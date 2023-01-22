import os
from typing import *
from db_models import DB_Model, DB_Csv, DB_Func, DB_Attack
import logging
from fabric import Connection
from config import HOST_IP, REMOTE_IP,REMOTE_USER,REMOTE_PORT,HOST_SSH_KEY
import os
from pathlib import Path
import threading
from hashlib import md5

pwd = os.path.split(os.path.realpath(__file__))[0]

def get_md5(filepath):
    file_hash = md5()
    with open(file=filepath, mode='rb') as f:
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

class DataManager(threading.Thread):
    def __init__(self, group = None, target: Callable[..., Any] = None, name: str = None, args: Iterable[Any] = ..., kwargs: Mapping[str, Any] = None , *, daemon: bool = None) -> None:
        super().__init__(None, target, name, args, kwargs, daemon=daemon)
        self.logger = logging.getLogger('DataManager')

    def run(self) -> None:
        self.query_and_transfer()
   
    def download(self, connection, remote, local, checksum, retry = 1):
        self.logger.info(f'Start downloading file :{remote}')
        transfer_res = connection.get(remote=remote, local=local)
        retry_cnt = 0
        check_successful = (get_md5(transfer_res.local) == checksum)
        while((not check_successful) and retry_cnt<retry):
            retry_cnt += 1
            self.logger.warn(f"File checksum inconsistent: {transfer_res.local} ")
            self.logger.warn(f'Retry times {retry_cnt}/{retry} .')
            transfer_res = connection.get(remote=remote, local=local)
            check_successful = (get_md5(transfer_res.local) == checksum)

        if(check_successful):    
            self.logger.info(f'Finish download file : {transfer_res.local}')
            return transfer_res.local
        else:
            self.logger.error(f'File checksum inconsistent after {retry} retry, delete file : {transfer_res.local}')
            os.remove(transfer_res.local) 
            return None

    def query_and_transfer(self):
        # acquire remote records in database
        waiting_list = []
        for download in [DB_Model, DB_Csv, DB_Attack]:
            list = download.select().where(download.host_ip != HOST_IP)
            waiting_list.extend(list)

        remain = len(waiting_list)
        if(len(waiting_list) == 0):
            return
        self.logger.info(f"Has {remain} records to download.")

        c = Connection(host=REMOTE_IP, user=REMOTE_USER, port=REMOTE_PORT, connect_kwargs={
            'key_filename':HOST_SSH_KEY
        })
        
        for i,entity in enumerate(waiting_list) :
            self.logger.info(f'Start process ({i+1}/{len(waiting_list)} record.)')
            loc = entity.location
            
            HEAD_205 = '/data2/'
            HEAD_139 = '/home/weichengkun/'

            if(loc.find(HEAD_139) != -1):
                # download from 139 to 205
                new_loc = loc.replace(HEAD_139,HEAD_205)
            else:
                # download from 205 to 139
                new_loc = loc.replace(HEAD_205,HEAD_139)

            dir,_ = os.path.split(new_loc)
            Path(dir).mkdir(parents=True, exist_ok=True)
            
            # download data from remote server
            dst_path = self.download(c, loc, dir+'/', entity.checksum,)

            # replace location with local path
            if(dst_path != None):
                entity.location = dst_path
                entity.host_ip = HOST_IP
            
                # update database
                entity.save()
                self.logger.info(f'Finish updating database')
            else:
                self.logger.error('Data download failed.')
            remain -= 1

        c.close()

# for single test
def main():
    t = DataManager()
    t.start()
    t.join()

if __name__ == "__main__":
    main()