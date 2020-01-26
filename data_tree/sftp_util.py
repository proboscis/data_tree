from urllib.parse import urlparse, ParseResult

import paramiko


class AutoSCPFile:
    def __init__(self,original_url,local_path):
        self.original_url = original_url
        self.local_path = local_path

    def download(self):
        res:ParseResult = urlparse(self.original_url)
        with paramiko.SSHClient() as sshc:
            sshc.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            split = res.netloc.split(":")
            host = split[0]
            if len(split) == 1:
                port = 22
            else:
                port = split[1]
            sshc.connect(hostname=host, port=port)
