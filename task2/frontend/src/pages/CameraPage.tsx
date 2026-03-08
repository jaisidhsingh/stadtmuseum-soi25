import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";

const CameraPage = () => {
  const navigate = useNavigate();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = React.useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [countdown, setCountdown] = React.useState<number | null>(null);

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

  // Timer Effect
  React.useEffect(() => {
    if (countdown === null) return;
    if (countdown === 0) {
      capturePhoto();
      setCountdown(null);
      return;
    }
    const timer = setTimeout(() => {
      setCountdown(countdown - 1);
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

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
    <div className="min-h-screen bg-background relative flex flex-col p-4 md:p-8">
      {/* Top Bar with Back Button */}
      <div className="absolute top-4 left-4 md:top-8 md:left-8 z-10">
        <Button
          variant="outline"
          size="lg"
          className="h-14 px-6 text-lg rounded-xl shadow-sm hover:shadow-md transition-all flex items-center gap-2 bg-background/80 backdrop-blur-sm"
          onClick={() => {
            stopCamera();
            navigate("/");
          }}
        >
          <ArrowLeft className="w-6 h-6" /> Back
        </Button>
      </div>

      <div className="flex-1 flex flex-col pt-16 md:pt-4">
        <h1 className="text-4xl font-bold text-center mb-4">Step 1: Capture Your Photo</h1>
        <p className="text-center text-xl text-primary font-medium mb-8 animate-pulse">
          Make sure your entire body (head to toes) is visible in the frame!
        </p>

        <div className="max-w-4xl mx-auto w-full flex-1">
          <Card className="shadow-lg border-2">
            <CardContent className="p-6">
              {error ? (
                <div className="text-center text-destructive p-8">
                  <p>{error}</p>
                  <Button onClick={startCamera} className="mt-4">Try Again</Button>
                </div>
              ) : capturedImage ? (
                <div className="space-y-6">
                  <img src={capturedImage} alt="Captured" className="w-full h-auto max-h-[60vh] object-contain rounded-lg bg-muted" />
                  <div className="flex justify-center gap-6">
                    <Button variant="outline" size="lg" className="h-16 px-8 text-xl" onClick={retakePhoto}>Retake Photo</Button>
                    <Button size="lg" className="h-16 px-12 text-xl shadow-md hover:shadow-lg transition-transform hover:scale-105" onClick={proceedToSelection}>Proceed to Styling</Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-6 relative">
                  <video ref={videoRef} autoPlay playsInline muted className="w-full h-auto max-h-[60vh] object-contain rounded-lg bg-muted border-2 border-primary/20" />

                  {countdown !== null && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="text-[150px] font-bold text-white drop-shadow-[0_4px_4px_rgba(0,0,0,0.8)] animate-pulse">
                        {countdown}
                      </div>
                    </div>
                  )}

                  <div className="flex justify-center">
                    {countdown === null ? (
                      <Button size="lg" className="h-20 px-16 text-2xl rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all bg-primary animate-in zoom-in duration-300" onClick={() => setCountdown(10)}>
                        📸 Start 10s Timer
                      </Button>
                    ) : (
                      <Button size="lg" variant="destructive" className="h-20 px-16 text-2xl rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all animate-in zoom-in duration-300" onClick={() => setCountdown(null)}>
                        Cancel Timer
                      </Button>
                    )}
                  </div>
                </div>
              )}
              <canvas ref={canvasRef} className="hidden" />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default CameraPage;
