import { ImageDownIcon, Upload } from "lucide-react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export  function ImagePreview() {
  return (
  
        <div className="grid gap-2 max-h-full">
          <img
            className="aspect-movie w-full rounded-md object-cover"
            src="/sign.jpeg"
          />
          <div className="grid grid-cols-3 gap-2">
            <button>
              <img
                alt="Product image"
                className="aspect-square w-full rounded-md object-cover"
                height="84"
                src="/sign.jpeg"
                width="84"
              />
            </button>
            <button>
              <img
                alt="Product image"
                className="aspect-square w-full rounded-md object-cover"
                height="84"
                src="/placeholder.svg"
                width="84"
              />
            </button>
            <button className="flex aspect-square w-full items-center justify-center rounded-md border border-dashed">
              <Upload className="h-4 w-4 text-muted-foreground" />
              <span className="sr-only">Upload</span>
            </button>
          </div>
        </div>
     
  )
}
