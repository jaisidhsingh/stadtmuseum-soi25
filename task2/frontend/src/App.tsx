import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import StartPage from "./pages/StartPage";
import CameraPage from "./pages/CameraPage";
import SelectionPage from "./pages/SelectionPage";
import ConfirmationPage from "./pages/ConfirmationPage";
import PrivacyFooter from "./components/PrivacyFooter";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const AppContent = () => {
  const location = useLocation();
  const showFooter = location.pathname !== "/";

  return (
    <>
      <Routes>
        <Route path="/" element={<StartPage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/selection" element={<SelectionPage />} />
        <Route path="/confirmation" element={<ConfirmationPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      {showFooter && <PrivacyFooter />}
    </>
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
