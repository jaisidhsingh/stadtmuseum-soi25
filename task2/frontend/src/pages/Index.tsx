// Update this page (the content is just a fallback if you fail to update the page)
import { t } from "@/lib/localization";

const Index = () => {
  return (
    <div className="flex h-full min-h-0 items-center justify-center bg-background">
      <div className="text-center">
        <h1 className="mb-4 text-4xl font-bold">
          {t("Welcome to Your Blank App", "Willkommen in deiner leeren App")}
        </h1>
        <p className="text-xl text-muted-foreground">
          {t(
            "Start building your amazing project here!",
            "Beginne hier mit deinem tollen Projekt!",
          )}
        </p>
      </div>
    </div>
  );
};

export default Index;
