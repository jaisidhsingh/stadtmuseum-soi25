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
import PrivacyFooter from "./components/PrivacyFooter";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const AppContent = () => {
  // Show footer on all pages
  const showFooter = true;

  return (
    <div className="app-shell min-h-screen exhibit-shell flex flex-col overflow-hidden">
      <main className="flex-1 min-h-0">
        <Routes>
          <Route path="/" element={<StartPage />} />
          <Route path="/camera" element={<CameraPage />} />
          <Route path="/test-input" element={<TestInputPage />} />
          <Route path="/silhouette" element={<SilhouettePage />} />
          <Route path="/selection" element={<SelectionPage />} />
          <Route path="/confirmation" element={<ConfirmationPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      {showFooter && <PrivacyFooter />}
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
