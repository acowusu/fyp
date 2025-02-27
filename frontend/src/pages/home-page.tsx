import { Button } from "@/components/ui/button";

export function Home() {
    return (
      
    <div
      className="flex flex-1 items-center justify-center rounded-lg border border-dashed shadow-sm"
    >
      <div className="flex flex-col items-center gap-1 text-center">
        <h3 className="text-2xl font-bold tracking-tight">
          You have no active Investigations
        </h3>
        <p className="text-sm text-muted-foreground">
          You can start an investigation by uploading an image.
        </p>
        <Button className="mt-4">Start Investigation</Button>
      </div>
    </div>


    );
  }