retz-client list-files -i $1
retz-client get-file -i $1 --path images.zip.base64 -R .
base64 -d images.zip.base64 > images.zip
