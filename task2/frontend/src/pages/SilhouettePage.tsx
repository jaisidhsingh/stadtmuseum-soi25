import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card } from "@/components/ui/card";
import { toast } from "@/hooks/use-toast";

type ImageResource = {
    id: string;
    url: string;
};

const SilhouettePage = () => {
    const navigate = useNavigate();
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const [previewImage, setPreviewImage] = useState<string | null>(null);
    const [previewResource, setPreviewResource] = useState<ImageResource | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    // Using an object to store styles
    const [styles, setStyles] = useState({
        arms: false,
        legs: false,
        head: false,
        feet: false,
    });

    // Add ref to prevent repeated segmentation of the same selections
    const lastRequestStyles = useRef<string | null>(null);

    useEffect(() => {
        const image = sessionStorage.getItem("capturedImage");
        if (image) {
            setCapturedImage(image);
        } else {
            // Redirect back to camera if no image found
            navigate("/camera");
        }
    }, [navigate]);

    // Helper function to convert dataURL to File
    const dataURLtoFile = (dataurl: string, filename: string) => {
        const arr = dataurl.split(',');
        const mimeMatch = arr[0].match(/:(.*?);/);
        const mime = mimeMatch ? mimeMatch[1] : '';
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, { type: mime });
    };

    // Effect to run live preview from backend endpoint
    useEffect(() => {
        if (!capturedImage) return;

        const selectedStyles = Object.entries(styles)
            .filter(([_, isSelected]) => isSelected)
            .map(([key]) => key)
            .sort()
            .join(',');

        if (lastRequestStyles.current === selectedStyles && previewImage) {
            return;
        }

        const runSegmentation = async () => {
            setIsProcessing(true);
            lastRequestStyles.current = selectedStyles;

            try {
                const formData = new FormData();
                const file = dataURLtoFile(capturedImage, 'capture.png');
                formData.append('image', file);

                if (selectedStyles.length > 0) {
                    formData.append('use_classical_warping', 'true');
                    formData.append('parts_to_warp', selectedStyles);
                } else {
                    formData.append('use_classical_warping', 'false');
                }

                const res = await fetch('http://localhost:8000/segment', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    throw new Error("Failed to process segment");
                }

                const data = await res.json();

                const finalImage = (data.stylized && data.stylized.length > 0)
                    ? data.stylized[0]
                    : data.original;

                setPreviewImage(`http://localhost:8000${finalImage.url}`);
                setPreviewResource(finalImage);

            } catch (error) {
                console.error("Error during segmentation:", error);
                toast({
                    title: "Processing Failed",
                    description: "Make sure the backend is running.",
                    variant: "destructive"
                });
                // Remove the failed styles to allow a retry
                lastRequestStyles.current = null;
            } finally {
                setIsProcessing(false);
            }
        };

        const timeoutId = setTimeout(() => {
            runSegmentation();
        }, 300); // 300ms debounce

        return () => clearTimeout(timeoutId);
    }, [capturedImage, styles, previewImage]);

    const handleStyleChange = (key: keyof typeof styles) => {
        setStyles((prev) => ({
            ...prev,
            [key]: !prev[key],
        }));
    };

    const handleNext = () => {
        if (previewResource) {
            sessionStorage.setItem("silhouetteData", JSON.stringify(previewResource));
            sessionStorage.setItem("silhouetteStyles", JSON.stringify(styles));
            navigate("/selection");
        }
    };

    return (
        <div className="min-h-screen bg-background p-8 flex flex-col md:flex-row gap-8">
            {/* Left Half: Large Captured Image / Preview */}
            <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-lg p-4 relative overflow-hidden">
                {isProcessing && (
                    <div className="absolute inset-0 bg-background/50 flex flex-col items-center justify-center z-10 backdrop-blur-sm">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                        <p className="text-lg font-medium">Processing...</p>
                    </div>
                )}

                {previewImage ? (
                    <img
                        src={previewImage}
                        alt="Segmented Silhouette"
                        className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg"
                    />
                ) : capturedImage ? (
                    <img
                        src={capturedImage}
                        alt="Captured Silhouette"
                        className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg opacity-50 grayscale"
                    />
                ) : (
                    <p className="text-muted-foreground">Loading image...</p>
                )}
            </div>

            {/* Right Half: Style Options */}
            <div className="flex-1 flex flex-col justify-center max-w-md mx-auto w-full">
                <Card className="p-8">
                    <h2 className="text-2xl font-bold mb-6 text-center">Add Style</h2>

                    <div className="space-y-6 mb-8">
                        {['arms', 'legs', 'head', 'feet'].map((part) => (
                            <div key={part} className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                                <Checkbox
                                    id={part}
                                    checked={styles[part as keyof typeof styles]}
                                    onCheckedChange={() => handleStyleChange(part as keyof typeof styles)}
                                    disabled={isProcessing}
                                />
                                <label
                                    htmlFor={part}
                                    className="text-lg font-medium capitalize leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                                >
                                    {part}
                                </label>
                            </div>
                        ))}
                    </div>

                    <div className="flex justify-end">
                        <Button
                            size="lg"
                            className="w-full text-lg py-6"
                            onClick={handleNext}
                            disabled={!previewResource || isProcessing}
                        >
                            Next
                        </Button>
                    </div>
                </Card>
            </div>
        </div>
    );
};

export default SilhouettePage;
