import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import dummyImage from "../assets/dummy.jpeg";

const TestInputPage = () => {
    const navigate = useNavigate();
    const [imageBase64, setImageBase64] = React.useState<string | null>(null);

    React.useEffect(() => {
        // Convert the imported image URL to base64 so it mimics the camera capture format
        const fetchImageAsBase64 = async () => {
            try {
                const response = await fetch(dummyImage);
                const blob = await response.blob();
                const reader = new FileReader();
                reader.onloadend = () => {
                    if (typeof reader.result === 'string') {
                        setImageBase64(reader.result);
                    }
                };
                reader.readAsDataURL(blob);
            } catch (error) {
                console.error("Failed to load test image as base64:", error);
            }
        };

        fetchImageAsBase64();
    }, []);

    const proceedToSelection = () => {
        if (imageBase64) {
            sessionStorage.setItem("capturedImage", imageBase64);
            navigate("/silhouette");
        }
    };

    return (
        <div className="min-h-screen bg-background p-8">
            <h1 className="text-2xl font-bold text-center mb-4">Test Input</h1>
            <p className="text-center text-muted-foreground mb-8">
                Using predefined test image instead of camera.
            </p>

            <div className="max-w-2xl mx-auto">
                <Card>
                    <CardContent className="p-6">
                        <div className="space-y-4">
                            <img src={dummyImage} alt="Test Input" className="w-full rounded-lg" />
                            <div className="flex justify-center gap-4">
                                <Button
                                    onClick={proceedToSelection}
                                    disabled={!imageBase64}
                                >
                                    Proceed with Test Image
                                </Button>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default TestInputPage;
