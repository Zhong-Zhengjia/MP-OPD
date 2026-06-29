import urllib.request
import os

print("http_proxy:", os.environ.get("http_proxy"))
print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("all_proxy:", os.environ.get("all_proxy"))

proxy_handler = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_handler)
print("Opener built")
