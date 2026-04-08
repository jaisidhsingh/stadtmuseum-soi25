import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { t } from "@/lib/localization";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname,
    );
  }, [location.pathname]);

  return (
    <div className="flex h-full min-h-0 items-center justify-center bg-muted">
      <div className="text-center">
        <h1 className="mb-4 text-4xl font-bold">404</h1>
        <p className="mb-4 text-xl text-muted-foreground">
          {t("Oops! Page not found", "Hoppla! Seite nicht gefunden")}
        </p>
        <a href="/" className="text-primary underline hover:text-primary/90">
          {t("Return to Home", "Zurueck zur Startseite")}
        </a>
      </div>
    </div>
  );
};

export default NotFound;
