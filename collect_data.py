import urllib.request as urlrequest #http requests
import urllib.parse as urlparse #parsing http parameters
import sys #command-line arguments
import json #to make rainwave data useable
import time #converting epoch time to dates

def main():
    # Check, then parse command line arguments.
    if len(sys.argv) < 3:
        print('Usage: python collect_data <user_id> <key>')
        return
    
    my_id = int(sys.argv[1])
    my_key = sys.argv[2]
    
    # General constant values
    sid_dict = { 'OCR' : 2 , 'ALL' : 5 }
    
    # Get information
    response = getInfo(my_id, my_key, sid_dict['ALL'])
    data = json.loads(response.read().decode('utf-8'))
    
    #Pull out important bits
    current = data['sched_current']
    
    # It's useless to take stats when voting's not allowed.
    if not current['voting_allowed']:
        return
        #contine #in the future, we will want to just skip this iteration

    #TODO: remove the information for printing.
    
    
    '''
    Print info about station, time, who listeners, previous election's number of
    voters, number of songs in the election, voter average over the past 10 min,
    and previous election winner's average rating, and if election was a tie*
    *without a request in the tie(??)
    '''
    
    '''
    Print info for each song: title, cooldown group, album, average rating,
    average rating for its album, request (0 or 1), requester's name,
    origin SID (if applicable), length, winner or loser, did it tie with winner?
    All 'string-like' things should be string + IDs when possible/reasonable.
    '''
    
    print(data)

def getInfo(my_id, my_key, station):
    """
    Perform a GET call to /api/info.

    Gets info about the currently playing song and returns it immediately.

    param: my_id,   a positive integer ID for the user
    param: my_key,  a string API key for the user
    param: station, a positive integer indicating the station (1-5)
    return:         the raw response (bytestream)  
    """
    
    #Put the parameters into the URL to allow for a GET.
    info_url = 'http://rainwave.cc/api4/info'
    params = { 'id' : my_id,
               'api_key' : my_key,
               'sid' : station}
    full_url = info_url + '?' + urlparse.urlencode(params)
    
    #Make the http request.
    return urlrequest.urlopen(full_url)

def postSync(my_id, my_key, station):
    """
    Perform a GET call to /api/info.

    Gets info about the currently playing song and returns it immediately.

    param: my_id,   a positive integer ID for the user
    param: my_key,  a string API key for the user
    param: station, a positive integer indicating the station (1-5)
    return:         the raw response (bytestream)  
    """
    
    #Params for syncing
    params = { 'user_id' : my_id,
               'key' : my_key,
               'sid' : station}
    data = urlparse.urlencode(params)
    data = data.encode('utf-8')
    
    #Perform sync
    sync_url = 'http://rainwave.cc/api4/sync'
    return urlrequest.urlopen(sync_url, data)


main()
