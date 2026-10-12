"""Verified official EasyOCR release assets through the GitHub API."""
import hashlib,json,shutil,time,urllib.request,zipfile
from urllib.parse import urlparse
from pathlib import Path
from run_record import sha

def ensure_weight(config,weights,output):
    target=weights/config['filename'];url=config['url'];route='verified-existing-cache'
    if not target.exists():
        parsed=urlparse(url);parts=parsed.path.strip('/').split('/')
        if parsed.scheme!='https' or parsed.netloc!='github.com' or len(parts)!=6 or parts[:4]!=['JaidedAI','EasyOCR','releases','download']:
            raise ValueError('Expected official JaidedAI/EasyOCR release asset')
        tag,name=parts[4:];api='https://api.github.com/repos/JaidedAI/EasyOCR/releases/tags/'+tag
        request=urllib.request.Request(api,headers={'User-Agent':'document-vqa-cuda'})
        with urllib.request.urlopen(request,timeout=25) as response:metadata=json.load(response)
        (output/(target.name+'-release.json')).write_text(json.dumps(metadata,indent=2))
        matches=[a for a in metadata['assets'] if a['name']==name]
        if len(matches)!=1:raise ValueError('Official release asset mismatch')
        asset=matches[0]
        if not asset['url'].startswith('https://api.github.com/repos/JaidedAI/EasyOCR/releases/assets/'):raise ValueError('Unexpected asset API')
        archive=output/(target.name+'.zip');partial=archive.with_suffix('.zip.partial')
        request=urllib.request.Request(asset['url']+'?download=1&time='+str(int(time.time())),headers={'Accept':'application/octet-stream','User-Agent':'document-vqa-cuda'})
        print('OCR_OFFICIAL_ASSET_DOWNLOADING',name,flush=True)
        with urllib.request.urlopen(request,timeout=45) as response,partial.open('xb') as destination:shutil.copyfileobj(response,destination)
        if partial.stat().st_size!=asset['size']:raise ValueError('Official asset size mismatch')
        partial.replace(archive)
        with zipfile.ZipFile(archive) as zipped:
            members=[n for n in zipped.namelist() if Path(n).name==target.name]
            if len(members)!=1:raise ValueError('Unexpected OCR model archive')
            data=zipped.read(members[0])
        if hashlib.md5(data).hexdigest()!=config['md5sum']:raise ValueError('Official OCR weight MD5 mismatch')
        target.write_bytes(data);route='official-github-release-api'
    if hashlib.md5(target.read_bytes()).hexdigest()!=config['md5sum']:raise ValueError('Official OCR weight MD5 mismatch')
    return {'url':url,'filename':target.name,'official_md5':config['md5sum'],'sha256':sha(target),'download_source':route}
