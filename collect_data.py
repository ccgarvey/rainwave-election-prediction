import urllib.request as urlrequest
import urllib.parse as urlparse
import sys

def main():
    # Check, then parse command line arguments.
    if len(sys.argv) < 3:
        print('Usage: python collect_data <user_id> <key>')
        return
    
    my_id = int(sys.argv[1])
    my_key = sys.argv[2]
    
    # General constant values
    sid_dict = { 'OCR' : 2 }
    
    #Params for syncing
    params = { 'user_id' : my_id,
               'key' : my_key,
               'sid' : sid_dict['OCR']}
    data = urlparse.urlencode(params)
    data = data.encode('utf-8')
    
    #Perform sync
    sync_url = 'http://rainwave.cc/api4/sync'
    response = urlrequest.urlopen(sync_url, data)
    print(response.read().decode('utf-8'))

"""
I'm keeping this example code because I may wish to use it in the future.
"""
def getParamsExample():
    info_url = 'http://rainwave.cc/api4/info'
    params = { 'id' : my_id,
               'api_key' : my_key,
               'sid' : sid_dict['OCR']}
    full_url = info_url + '?' + urlparse.urlencode(params)

main()
