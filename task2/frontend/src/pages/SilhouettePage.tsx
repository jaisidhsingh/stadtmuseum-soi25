import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card } from "@/components/ui/card";

const SilhouettePage = () => {
    const navigate = useNavigate();
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const [styles, setStyles] = useState({
        arms: false,
        legs: false,
        head: false,
        feet: false,
    });

    useEffect(() => {
        const image = sessionStorage.getItem("capturedImage");
        if (image) {
            setCapturedImage(image);
        } else {
            // Redirect back to camera if no image found
            navigate("/camera");
        }
    }, [navigate]);

    const handleStyleChange = (key: keyof typeof styles) => {
        setStyles((prev) => ({
            ...prev,
            [key]: !prev[key],
        }));
    };

    const handleNext = () => {
        sessionStorage.setItem("silhouetteStyles", JSON.stringify(styles));
        navigate("/selection");
    };

    return (
        <div className="min-h-screen bg-background p-8 flex flex-col md:flex-row gap-8">
            {/* Left Half: Large Captured Image */}
            <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-lg p-4">
                {capturedImage ? (
                    <img
                        src={capturedImage}
                        alt="Captured Silhouette"
                        className="max-h-[80vh] w-auto object-contain rounded-lg shadow-lg"
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
                        <div className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                            <Checkbox
                                id="arms"
                                checked={styles.arms}
                                onCheckedChange={() => handleStyleChange("arms")}
                            />
                            <label
                                htmlFor="arms"
                                className="text-lg font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                            >
                                Arms
                            </label>
                        </div>

                        <div className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                            <Checkbox
                                id="legs"
                                checked={styles.legs}
                                onCheckedChange={() => handleStyleChange("legs")}
                            />
                            <label
                                htmlFor="legs"
                                className="text-lg font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                            >
                                Legs
                            </label>
                        </div>

                        <div className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                            <Checkbox
                                id="head"
                                checked={styles.head}
                                onCheckedChange={() => handleStyleChange("head")}
                            />
                            <label
                                htmlFor="head"
                                className="text-lg font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                            >
                                Head
                            </label>
                        </div>

                        <div className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                            <Checkbox
                                id="feet"
                                checked={styles.feet}
                                onCheckedChange={() => handleStyleChange("feet")}
                            />
                            <label
                                htmlFor="feet"
                                className="text-lg font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer flex-1"
                            >
                                Feet
                            </label>
                        </div>
                    </div>

                    <div className="flex justify-end">
                        <Button
                            size="lg"
                            className="w-full text-lg py-6"
                            onClick={handleNext}
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
