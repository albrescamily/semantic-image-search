from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.services.storage_service import upload_to_s3
from app.services.embedding_service import embed_image_from_s3,embed_text_query
from app.services.indexing_service import index_image
from app.services.ann_service import ann_search 
from core.clients import BUCKET, s3_client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Server is running smoothly!"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    bucket, key = upload_to_s3(file)
    embedding = embed_image_from_s3(bucket, key)
    index_image(bucket, key, embedding)

    return {"message": "Uploaded successfully!"}


@app.post("/query_text")
async def input_query_text(query: str):

    query_embed = embed_text_query(query)
    results = ann_search(query_embed)

    images = [
        {
            "key": payload["key"],
            "url": s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": payload["bucket"], "Key": payload["key"]},
                ExpiresIn=3600,
            ),
        }
        for payload in results
    ]

    return {"images": images}



@app.get("/images")
async def list_images():
    response = s3_client.list_objects_v2(Bucket=BUCKET)

    images = [
        {
            "key": obj["Key"],
            "url": s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET, "Key": obj["Key"]},
                ExpiresIn=3600,
            ),
        }
        for obj in response.get("Contents", [])
    ]

    return {"images": images}


@app.delete("/delete-image")
async def delete_image(key: str):
    s3_client.delete_object(
        Bucket=BUCKET,
        Key=key
    )

    return {"message": "Deleted successfully!"}
