import base64
import datetime

def compress(data):
    log_tool_use("compress", data)
    return f"COMPRESSED({data})"

def encode_base64(data):
    log_tool_use("encode_base64", data)
    return base64.b64encode(data.encode()).decode()

def cloud_store(bucket_name, data):
    log_tool_use("cloud_store", f"bucket: {bucket_name}, data: {data[:50]}...")
    filename = f"{bucket_name}_output.txt"
    with open(filename, "w") as f:
        f.write(data)
    return filename

def log_tool_use(tool_name, data):
    with open("log.txt", "a") as log:
        log.write(f"[{datetime.datetime.now()}] TOOL USED: {tool_name} | Data: {data}\n")
