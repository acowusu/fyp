import {  Upload } from "lucide-react"



export  function ImagePreview() {
  return (
  
        <div className="grid gap-2 max-h-48">
          <img
            className="aspect-movie  rounded-md object-scale-down"
            src="/sign.jpeg"
          />
          <div className="grid grid-cols-3 gap-2">
            <button>
              <img
                alt="Product image"
                className="aspect-square  rounded-md object-scale-down"
                src="/sign.jpeg"
              />
            </button>
            <button>
              <img
                alt="Product image"
                className="aspect-square  rounded-md object-scale-down"
                src="/placeholder.svg"
              />
            </button>
            <button className="flex aspect-square items-center justify-center rounded-md border border-dashed">
              <Upload className="h-4 w-4 text-muted-foreground" />
              <span className="sr-only">Upload</span>
            </button>
          </div>
        </div>
     
  )
}
