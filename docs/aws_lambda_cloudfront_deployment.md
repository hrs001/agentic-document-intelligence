# AWS Deployment Guide

This project is prepared for:

- backend: `AWS Lambda + API Gateway`
- frontend: `S3 + CloudFront`
- runtime artifacts: `S3`

## Security notes

- Never upload `.env`.
- Pass only the required secret values to Lambda as environment variables at deploy time.
- The SAM template marks the API keys as `NoEcho`.
- Keep IAM permissions tight. The backend only needs `s3:GetObject` for the runtime bundle prefix.

## Deployment shape

The deployment uses three staged build outputs:

- `deployment/dist/backend-src`
  - minimal Lambda source tree
- `deployment/dist/runtime-bundle`
  - Kuzu, SQLite, corpus metadata, chunk artifacts, and demo files
- `deployment/dist/frontend-site`
  - static frontend export for S3/CloudFront

At Lambda cold start:

- the backend downloads the runtime bundle from S3 into `/tmp/adi_runtime`
- the app then reads Kuzu, SQLite, chunk metadata, and demo files from the hydrated local paths

## Build steps

From `agentic_document_intelligence`:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/build-backend-stage.ps1
powershell -ExecutionPolicy Bypass -File deployment/build-runtime-bundle.ps1
```

Build the static frontend export with the verified direct command:

```powershell
powershell -ExecutionPolicy Bypass -Command "cd frontend; npm.cmd run -s build"
powershell -ExecutionPolicy Bypass -File deployment/build-frontend-static.ps1
```

## Upload runtime bundle to S3

If you use the existing shared bucket, replace only the prefix as needed:

```powershell
aws s3 sync deployment/dist/runtime-bundle s3://rag-universal-knowledge/runtime-bundle/
```

## Build and deploy the backend stack

Use SAM with container build so native Python dependencies such as `kuzu` are built in a Lambda-compatible Linux environment:

```powershell
sam build -t template.yaml --use-container
sam deploy --guided
```

Important deploy-time parameters:

- `OpenAIApiKey`
- `PineconeApiKey`
- `PineconeIndexName`
- `RuntimeAssetsBucketName`
- `RuntimeAssetsPrefix`
- `FrontendAssetsBucketName`
- `FrontendAssetsPrefix`

Suggested first-pass Lambda sizing:

- memory: `10240`
- timeout: `900`
- ephemeral storage: `2048`

## Upload frontend assets

After the stack is created, upload the static frontend build to the chosen frontend prefix:

```powershell
aws s3 sync deployment/dist/frontend-site s3://rag-universal-knowledge/frontend-site/ --delete
```

The CloudFront distribution in the template serves:

- static frontend assets from the configured frontend bucket/prefix
- `/api/*` and `/health` from API Gateway

That means the frontend can use same-origin API calls in production.

## Validation checklist

Before public release, verify:

- `Load Demo Experience` works
- document and CSV download links work
- graph panel loads
- citations render correctly
- RAGAS metrics render
- complex mixed queries succeed
- follow-up conversation queries still work

## Notes

- If `kuzu` turns out to be problematic in zip packaging despite `sam build --use-container`, the fallback is a Lambda container image, not a change to the application logic.
- Runtime artifacts stay in S3 and are never bundled into the frontend.
