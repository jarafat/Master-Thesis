import requests

# URL to CompactTV videos
url = "https://tib.eu/cloud/s/RBMaSGR9DXEBp6E/download?path=%2FCompactTV&files=mp4_tv_compact_online_de&downloadStartSecret=8kooac0btff"
response = requests.get(url)

# Store in .zip
with open("/nfs/home/arafatj/CompactTVvideos/videos.zip", "wb") as f:
    f.write(response.content)