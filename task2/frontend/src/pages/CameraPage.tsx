import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const CameraPage = () => {
  const navigate = useNavigate();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = React.useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const startCamera = React.useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setError(null);
    } catch (err) {
      setError("Unable to access camera. Please grant camera permissions.");
      console.error("Camera error:", err);
    }
  }, []);

  const stopCamera = React.useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  }, [stream]);

  React.useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL("image/png");
        setCapturedImage(imageData);
        stopCamera();
      }
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    startCamera();
  };

  const proceedToSelection = () => {
    if (capturedImage) {
      sessionStorage.setItem("capturedImage", capturedImage);
      navigate("/silhouette");
    }
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="text-2xl font-bold text-center mb-4">Capture Your Photo</h1>
      <p className="text-center text-muted-foreground mb-8">
        Position yourself in the frame and click capture when ready
      </p>

      <div className="max-w-2xl mx-auto">
        <Card>
          <CardContent className="p-6">
            {error ? (
              <div className="text-center text-destructive p-8">
                <p>{error}</p>
                <Button onClick={startCamera} className="mt-4">Try Again</Button>
              </div>
            ) : capturedImage ? (
              <div className="space-y-4">
                <img src={capturedImage} alt="Captured" className="w-full rounded-lg" />
                <div className="flex justify-center gap-4">
                  <Button variant="outline" onClick={retakePhoto}>Retake</Button>
                  <Button onClick={proceedToSelection}>Proceed</Button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-lg bg-muted" />
                <div className="flex justify-center">
                  <Button size="lg" onClick={capturePhoto}>Capture</Button>
                </div>
              </div>
            )}
            <canvas ref={canvasRef} className="hidden" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default CameraPage;
