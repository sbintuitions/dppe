import argparse
import asyncio
import base64
import copy
import json
import os
import urllib.request
from urllib import parse

import quickxorhash
import requests

# Import Playwright
from playwright.async_api import async_playwright
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Simulate browser
header = {
    "sec-ch-ua-mobile": "?0",
    "upgrade-insecure-requests": "1",
    "dnt": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36 Edg/90.0.818.51",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "service-worker-navigation-preload": "true",
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "navigate",
    "sec-fetch-dest": "iframe",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
}


def parse_args():
    """
    Args:
        url: Dataset share link.
        pwd: Password of the share folder. You can contact the author for data access.
        download_root: Path to store downloaded data.
        force: Whether to force download data.
            Default value is False, assuming the user just wants to update data.
            If you have no data in the local path, please set this argument to True.
    """
    parse = argparse.ArgumentParser(description="Download MVImgNet data")
    parse.add_argument("--url", type=str, required=True, help="Dataset share link")
    parse.add_argument("--pwd", type=str, required=True, help="Password of the share folder")
    parse.add_argument(
        "--download_root",
        type=str,
        default="./dataset/raw/MVImgNet2",
        help="Path to store downloaded data",
    )
    parse.add_argument(
        "--force", action="store_true", default=False, help="Whether to force download data"
    )

    args = parse.parse_args()
    return args


def newSession():
    s = requests.session()
    retries = Retry(total=5, backoff_factor=0.1)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def save_hash(path, code):
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)


def read_hash(path):
    with open(path, "r", encoding="utf-8") as f:
        str = f.read()
    return str


def checkHashes(localfile, cloud_hash, localroot, force):
    if not os.path.exists(os.path.dirname(localfile)):
        os.makedirs(os.path.dirname(localfile), exist_ok=True)
    if localfile.split("/")[1] == "pretrained_models":
        local_hash_bkup = os.path.join(
            localroot,
            ".hash",
            os.path.join(localfile.split("/")[2], localfile.split("/")[3].split(".")[0] + ".txt"),
        )
    else:
        local_hash_bkup = os.path.join(
            localroot, ".hash", os.path.join(localfile.split("/")[-1].split(".")[0] + ".txt")
        )
    if not os.path.exists(os.path.dirname(local_hash_bkup)):
        os.makedirs(os.path.dirname(local_hash_bkup), exist_ok=True)

    if force:
        save_hash(local_hash_bkup, cloud_hash["quickXorHash"])
        tqdm.write("Force downloading data")
        return False

    if os.path.exists(localfile):
        with open(localfile, "rb") as lf:
            content = lf.read()
            hash = quickxorhash.quickxorhash()
            hash.update(content)
            hashoutput = base64.b64encode(hash.digest()).decode("ascii")
            save_hash(local_hash_bkup, cloud_hash["quickXorHash"])
            if hashoutput == cloud_hash["quickXorHash"]:
                tqdm.write(
                    f"[{os.path.relpath(localfile, localroot)}] Local file is up-to-date, skipping download"
                )
                return True
            else:
                tqdm.write(
                    f"[{os.path.relpath(localfile, localroot)}] Local file is out-of-date, updating"
                )
                return False
    else:
        if os.path.isfile(local_hash_bkup):
            hashoutput = read_hash(local_hash_bkup)
            if hashoutput == cloud_hash["quickXorHash"]:
                tqdm.write(
                    f"[{os.path.relpath(localfile, localroot)}] Local file is up-to-date, skipping download"
                )
                return True
            else:
                save_hash(local_hash_bkup, cloud_hash["quickXorHash"])
                tqdm.write(
                    f"[{os.path.relpath(localfile, localroot)}] Local file is out-of-date, updating"
                )
                return False
        else:
            save_hash(local_hash_bkup, cloud_hash["quickXorHash"])
            tqdm.write(
                f"[{os.path.basename(localfile)}] No local file or local file is missing, downloading"
            )
            return False


def getFiles(originalUrl, download_path, force, download_root=None, req=None, layers=0, _id=0):
    isSharepoint = False
    if "-my" not in originalUrl:
        isSharepoint = True
    if req is None:
        req = newSession()

    reqf = req.get(originalUrl, headers=header)

    # Bypass the strict check but notify the user
    if ',"FirstRow"' not in reqf.text:
        print(
            "[Warning] 'FirstRow' missing from HTML. Bypassing check and attempting API request..."
        )

    if download_root is None:
        download_root = download_path

    filesData = []
    redirectURL = reqf.url

    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(redirectURL).query))

    if "id" not in query:
        print("[Error] Failed to find folder ID in the URL. This might be a bot-protection page.")
        return 0

    redirectSplitURL = redirectURL.split("/")

    relativeFolder = ""
    rootFolder = query["id"]
    for i in rootFolder.split("/"):
        if isSharepoint:
            if i != "Shared Documents":
                relativeFolder += i + "/"
            else:
                relativeFolder += i
                break
        else:
            if i != "Documents":
                relativeFolder += i + "/"
            else:
                relativeFolder += i
                break

    relativeUrl = (
        parse.quote(relativeFolder).replace("/", "%2F").replace("_", "%5F").replace("-", "%2D")
    )
    rootFolderUrl = (
        parse.quote(rootFolder).replace("/", "%2F").replace("_", "%5F").replace("-", "%2D")
    )

    graphqlVar = (
        '{"query":"query (\n        $listServerRelativeUrl: String!,$renderListDataAsStreamParameters: RenderListDataAsStreamParameters!,$renderListDataAsStreamQueryString: String!\n        )\n      {\n      \n      legacy {\n      \n      renderListDataAsStream(\n      listServerRelativeUrl: $listServerRelativeUrl,\n      parameters: $renderListDataAsStreamParameters,\n      queryString: $renderListDataAsStreamQueryString\n      )\n    }\n      \n      \n  perf {\n    executionTime\n    overheadTime\n    parsingTime\n    queryCount\n    validationTime\n    resolvers {\n      name\n      queryCount\n      resolveTime\n      waitTime\n    }\n  }\n    }","variables":{"listServerRelativeUrl":"%s","renderListDataAsStreamParameters":{"renderOptions":5707527,"allowMultipleValueFilterForTaxonomyFields":true,"addRequiredFields":true,"folderServerRelativeUrl":"%s"},"renderListDataAsStreamQueryString":"@a1=\'%s\'&RootFolder=%s&TryNewExperienceSingle=TRUE"}}'
        % (relativeFolder, rootFolder, relativeUrl, rootFolderUrl)
    )

    s2 = urllib.parse.urlparse(redirectURL)
    tempHeader = copy.deepcopy(header)
    tempHeader["referer"] = redirectURL

    # Safe cookie merge: Combine Playwright cookies with any new session cookies
    current_cookie = header.get("cookie", "")
    new_cookie = reqf.headers.get("set-cookie", "")
    tempHeader["cookie"] = f"{current_cookie}; {new_cookie}" if new_cookie else current_cookie

    tempHeader["authority"] = s2.netloc
    tempHeader["content-type"] = "application/json;odata=verbose"

    # Send the actual GraphQL request to retrieve the files
    graphqlReq = req.post(
        "/".join(redirectSplitURL[:-3]) + "/_api/v2.1/graphql",
        data=graphqlVar.encode("utf-8"),
        headers=tempHeader,
    )

    try:
        graphqlReq_json = json.loads(graphqlReq.text)
    except json.JSONDecodeError:
        print("[Error] Failed to parse API response. SharePoint likely blocked the request.")
        return 0

    if "NextHref" in graphqlReq_json["data"]["legacy"]["renderListDataAsStream"]["ListData"]:
        nextHref = graphqlReq_json["data"]["legacy"]["renderListDataAsStream"]["ListData"][
            "NextHref"
        ] + "&@a1=%s&TryNewExperienceSingle=TRUE" % ("%27" + relativeUrl + "%27")
        filesData.extend(
            graphqlReq_json["data"]["legacy"]["renderListDataAsStream"]["ListData"]["Row"]
        )

        listViewXml = graphqlReq_json["data"]["legacy"]["renderListDataAsStream"]["ViewMetadata"][
            "ListViewXml"
        ]
        renderListDataAsStreamVar = (
            '{"parameters":{"__metadata":{"type":"SP.RenderListDataParameters"},"RenderOptions":1216519,"ViewXml":"%s","AllowMultipleValueFilterForTaxonomyFields":true,"AddRequiredFields":true}}'
            % (listViewXml).replace('"', '\\"')
        )

        graphqlReq = req.post(
            "/".join(redirectSplitURL[:-3])
            + "/_api/web/GetListUsingPath(DecodedUrl=@a1)/RenderListDataAsStream"
            + nextHref,
            data=renderListDataAsStreamVar.encode("utf-8"),
            headers=tempHeader,
        )
        graphqlReq_json = json.loads(graphqlReq.text)

        while "NextHref" in graphqlReq_json["ListData"]:
            nextHref = graphqlReq_json["ListData"][
                "NextHref"
            ] + "&@a1=%s&TryNewExperienceSingle=TRUE" % ("%27" + relativeUrl + "%27")
            filesData.extend(graphqlReq_json["ListData"]["Row"])
            graphqlReq = req.post(
                "/".join(redirectSplitURL[:-3])
                + "/_api/web/GetListUsingPath(DecodedUrl=@a1)/RenderListDataAsStream"
                + nextHref,
                data=renderListDataAsStreamVar.encode("utf-8"),
                headers=tempHeader,
            )
            graphqlReq_json = json.loads(graphqlReq.text)
        filesData.extend(graphqlReq_json["ListData"]["Row"])
    else:
        filesData.extend(
            graphqlReq_json["data"]["legacy"]["renderListDataAsStream"]["ListData"]["Row"]
        )

    filesData = sorted(filesData, key=lambda x: x["FileLeafRef"])
    for i in filesData:
        if i["FSObjType"] == "1":
            _query = query.copy()
            _query["id"] = os.path.join(_query["id"], i["FileLeafRef"]).replace("\\", "/")
            if not isSharepoint:
                originalPath = (
                    "/".join(redirectSplitURL[:-1])
                    + "/onedrive.aspx?"
                    + urllib.parse.urlencode(_query)
                )
            else:
                originalPath = (
                    "/".join(redirectSplitURL[:-1])
                    + "/AllItems.aspx?"
                    + urllib.parse.urlencode(_query)
                )
            getFiles(
                originalPath,
                os.path.join(download_path, i["FileLeafRef"]),
                force,
                download_root,
                req=req,
                layers=layers + 1,
            )
        else:
            reqf = req.get(i[".spItemUrl"], headers=header)
            filemeta = json.loads(reqf.text)

            url_download, name, hash_val = (
                filemeta["@content.downloadUrl"],
                filemeta["name"],
                filemeta["file"]["hashes"],
            )
            r = requests.get(url_download, stream=True)
            total_length = int(r.headers.get("content-length", 0))
            local_file = os.path.join(download_path, name)

            if not checkHashes(local_file, hash_val, download_root, force):
                with (
                    open(os.path.join(download_path, name), "wb") as f,
                    tqdm(
                        desc=os.path.relpath(local_file, download_root),
                        total=total_length,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)


pheader = ""
url = ""


async def fetch_with_pwd(iurl, password):
    """
    Perform password authentication using Playwright, retrieve cookies,
    and synchronize the User-Agent to prevent session blocking.
    """
    global pheader, url, header
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(iurl, wait_until="networkidle")
        await page.locator("input[id='txtPassword']").fill(password)
        print("Password input complete, jumping...")

        await page.locator("input[id='btnSubmitPassword']").click()
        print("Waiting for SharePoint redirects to settle...")

        await page.wait_for_timeout(5000)
        await page.wait_for_load_state("networkidle")

        url = page.url
        print(f"Landed on URL: {url}")

        ua = await page.evaluate("navigator.userAgent")
        header["user-agent"] = ua

        print("Fetching cookies...")
        _cookies = await context.cookies()

        pheader = "".join([f"{c['name']}={c['value']}; " for c in _cookies])

        await browser.close()


def havePwdGetFiles(iurl, password, download_path, force):
    global header
    asyncio.run(fetch_with_pwd(iurl, password))
    header["cookie"] = pheader
    getFiles(url, download_path, force)


if __name__ == "__main__":
    args = parse_args()
    url_target = args.url
    pwd = args.pwd
    download_root = args.download_root
    force = args.force

    havePwdGetFiles(url_target, pwd, download_root, force)
