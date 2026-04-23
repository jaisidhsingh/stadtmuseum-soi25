import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import StartPage from "./pages/StartPage";
import CameraPage from "./pages/CameraPage";
import SilhouettePage from "./pages/SilhouettePage";
import SelectionPage from "./pages/SelectionPage";
import ConfirmationPage from "./pages/ConfirmationPage";
import TestInputPage from "./pages/TestInputPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const AppContent = () => {
  return (
    <div className="app-shell flex h-svh min-h-0 max-h-svh flex-col overflow-hidden">
      <main className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <Routes>
            <Route path="/" element={<StartPage />} />
            <Route path="/camera" element={<CameraPage />} />
            <Route path="/test-input" element={<TestInputPage />} />
            <Route path="/silhouette" element={<SilhouettePage />} />
            <Route path="/selection" element={<SelectionPage />} />
            <Route path="/confirmation" element={<ConfirmationPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </main>
    </div>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
