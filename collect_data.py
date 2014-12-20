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
    
    # Retrieve listeners information (it's not given in sync)
    listeners_response = postListeners(my_id, my_key, sid_dict['OCR'])
    listeners = json.loads(listeners_response.read().decode('utf-8'))
    
    # Retrieve station information
    sid = sid_dict['OCR']
    response = getInfo(my_id, my_key, sid)
    data = json.loads(response.read().decode('utf-8'))

    # Pull out important bits
    current = data['sched_current']
    previous = data['sched_history']
    
    '''
    this does not work; voting is never allowed on current.
    Maybe check with a while loop?
    # It's useless to take stats when voting's not allowed.
    if not current['voting_allowed']:
        print('No voting.')
        return
        #contine #in the future, we will want to just skip this iteration
    '''
    epoch_time = current['start']
    n_songs = len(current['songs'])
    prev_votes = tallyVotes(previous[0])
    
    voter_avg = averageVotes(previous)
    prev_winner_rating = previous[0]['songs'][0]['rating']
    is_random = checkRandom(current)
    
    '''
    Get info about station, time, listeners, previous election's number of
    voters, number of songs in the election, voter average over past 5 songs,
    previous election winner's average rating, and if election was a tie*
    *without a request in the tie(??)
    '''
    election_out = { 'station' : sid,
                     'time' : epoch_time, #subject to change!
                     'number_songs' : n_songs,
                     'prev_elec_votes' : prev_votes,
                     'prev_5_elec_vote_avg' : voter_avg,
                     'prev_winner_rating' : prev_winner_rating,
                     'chosen_randomly' : is_random,
                     'listeners' : listeners,
                     'songs' : [] }
    
    '''
    Get info for each song: title, cooldown group, album, average rating,
    average rating for its album, was it a request or not, requester's name,
    origin SID (if applicable), length, did it play, did it tie with winner?,
    artists
    All 'string-like' things should be string + IDs when possible/reasonable.
    '''
    # get information that can be fetched from a single song
    for song in current['songs']:
        election_out['songs'].append(processSong(song))
    # TODO: did it tie with the winner + did it play?
    
    print(json.dumps(election_out, indent=4, sort_keys=True))

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
               'sid' : station }
    data = urlparse.urlencode(params)
    data = data.encode('utf-8')
    
    #Perform sync
    sync_url = 'http://rainwave.cc/api4/sync'
    return urlrequest.urlopen(sync_url, data)

def postListeners(my_id, my_key, station):
    '''
    Perform a POST call to /api/current_listeners.
    
    POST to /api/sync doesn't include this information.

    param: my_id,   a positive integer ID for the user
    param: my_key,  a string API key for the user
    param: station, a positive integer indicating the station (1-5)
    return:         the raw response (bytestream)  
    '''
    params = { 'user_id' : my_id,
               'key' : my_key,
               'sid' : station }
    data = urlparse.urlencode(params)
    data = data.encode('utf-8')
    
    #Perform sync
    listeners_url = 'http://rainwave.cc/api4/current_listeners'
    return urlrequest.urlopen(listeners_url, data)

def tallyVotes(election):
    '''
    Tallies the votes in an election.
    
    param: election,  a dictionary, in standard rainwave format, of an election
    return:           an integer count of the number of total votes
    '''
    songs = election['songs']
    total = 0
    
    for song in songs:
        total += song['entry_votes']
    
    return total

def averageVotes(elections):
    '''
    Averages the votes over all elections, presumably in previous elections.
    
    param: elections,  a dictionary containing several elections
    return:            a double number of votes, the average over all
    '''
    total = 0
    count = 0
    
    for entry in elections:
        for song in entry['songs']:
            total += song['entry_votes']
        count += 1
    
    return total/count

def checkTied(election):
    '''
    Determines if an election was a tie.
    
    param: election, the election to check (in standard rainwave format)
    return: True iff. the election was tied, regardless of requests
    return: number of songs tied
    '''
    songs = election['songs']
    
    # Single-song elections can't tie.
    if len(songs) <= 1:
        return false
    
    # Election is only tied if the top two songs have the same number of votes.
    n_tied = 1
    for i in range(0, len(songs)-1):
        if songs[i]['entry_votes'] == songs[i+1]['entry_votes']:
            n_tied += 1
        else:
            break
    
    return (n_tied > 1), n_tied
    
def checkRandom(election):
    '''
    Determines if the result of an election was chosen randomly.
    
    param: election, the election to check (in standard rainwave format)
    return:          True iff. the election was randomly chosen.
    '''
    songs = election['songs']
    
    # Check if there was a tie at all. No tie -> not random.
    # Also handles single-song elections. 
    is_tied = checkTied(election)[0]
    
    if not is_tied:
        return False
    
    # If the winner was a request, and the second wasn't, it wasn't random.
    # (handles special case: 
    elif requestInfo(songs[0])[0] and not requestInfo(songs[1])[0]:
        return False
    else:
        return True

def requestInfo(song):
    '''
    Returns information about whether or not a request was made on a song, and,
    if so, who requested it.
    
    param: song,  the song object (in standard rainwave foramt)
    return:       True iff. the song was requested, False otherwise
    return:       the name of the requester, or None if there was no request
    '''
    name = song['elec_request_username']
    if name == None:
        return False, None
    else:
        return True, name

main()
