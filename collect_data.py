import urllib.request as urlrequest #http requests
import urllib.parse as urlparse #parsing http parameters
import urllib.error
import sys #command-line arguments
import json #to make rainwave data useable
import time #converting epoch time to dates


""" ****************************************************************************
                            DATA COLLECTION AND WRITING
**************************************************************************** """
def main():
    # Check, then parse command line arguments.
    if len(sys.argv) != 4:
        print('Usage: python collect_data <user_id> <key> <station>')
        return
    
    my_id = int(sys.argv[1])
    my_key = sys.argv[2]
    station = sys.argv[3]
    
    # General constant values
    sid_dict = { 'GAME' : 1, 'OCR' : 2, 'COVERS' : 3, 'CHIP' : 4, 'ALL' : 5}
    station_dict = dict(reversed(i) for i in sid_dict.items())
    
    # Station selection
    if station in sid_dict:
        sid = sid_dict[station]
    else:
        sid = int(station)
    
    # Time keeping
    time_of_day = time.localtime()
    current_day = time_of_day.tm_wday
    #start_day = time_of_day.tm_wday
    #start_month = time_of_day.tm_mon
    #start_year = time_of_day.tm_year
    first_run = True
    collected = 0
    
    #current_day = start_day
    now_day = current_day
    
    while True:
        time_of_day = time.localtime()
        now_day = time_of_day.tm_wday
        
        # Day has rolled over
        if now_day != current_day or first_run:
            collected = 0
            current_day = now_day
            
            # If this is not the first go-around, open files should be closed.
            if not first_run:
                data_file.write('\n]\n')
                data_file.close()
                data_file.close()
            else:
                first_run = False
            
            current_month = time_of_day.tm_mon
            current_year = time_of_day.tm_year
            current_mday = time_of_day.tm_mday
            
            # Files: data, and error logging
            out_file_name = (station_dict[sid] + '_' + str(current_month) + '_'
                             + str(current_mday) + '_' + str(current_year))
            log_file_name = (out_file_name + '.log')
            data_file_name = (out_file_name + '.data')
            log_file = open(log_file_name, 'a')
            data_file = open(data_file_name, 'a')
    
            # Begin an array in the data file
            data_file.write('[')
            
        # Attempt HTTP requests.
        try:
            # Retrieve listeners information (it's not given in sync)
            listener_response = postListeners(my_id, my_key, sid)
            listeners = json.loads(listener_response.read().decode('utf-8'))
            
            # Retrieve station information (involves HTTP request)
            response = postSync(my_id, my_key, sid)
            
            print('Response received')
            
            data = json.loads(response.read().decode('utf-8'))

            # Pull out important bits
            current = data['sched_current']
            previous = data['sched_history']
            
            epoch_time = current['start_actual'] 
            # This is when it the election results played, NOT 'start' nor 'end'
            
            n_songs = len(current['songs'])
            prev_votes = tallyVotes(previous[0])
            
            voter_avg = averageVotes(previous)
            prev_winner_rating = previous[0]['songs'][0]['rating']
            is_random = checkRandom(current)
            
            '''
            Get info about station, time, listeners, previous election's number
            of voters, number of songs in the election, voter average over past
            5 songs, previous election winner's average rating, and if election
            was a tie.
            '''
            election_out = { 'station' : sid,
                            'time' : parseTime(epoch_time),
                            'number_songs' : n_songs,
                            'prev_elec_votes' : prev_votes,
                            'prev_5_elec_vote_avg' : voter_avg,
                            'prev_winner_rating' : prev_winner_rating,
                            'chosen_randomly' : is_random,
                            'listeners' : listeners,
                            'songs' : [] }
            
            '''
            Get info for each song: title, ID, cooldown group, album, average
            rating, average rating for its album, was it a request or not, 
            requester's name, origin SID, length, did it play, did it tie with
            winner?, artists
            '''
            # get information that can be fetched from a single song
            has_tie, n_tied = checkTied(current)
            for i in range(len(current['songs'])):
                processed_song = processSong(current['songs'][i])
                processed_song['tied_winner'] = (i < n_tied)
                processed_song['was_played'] = (i == 0)
                election_out['songs'].append(processed_song)
            
            if collected > 0:
                data_file.write(',')
            data_file.write('\n')
            data_file.write(json.dumps(election_out, indent=4, sort_keys=True))
            collected += 1
            
            print('Collected a song in the log.')
        
        # If either fails, log the error, and wait for 60 seconds.
        except urllib.error.URLError as error:
            logExceptionAndWait(log_file, error.reason, 60)
            continue
        
        except Exception as error:
            print(data)
            logExceptionAndWait(log_file, str(error))
            continue
    
    # End the array in the data file
    data_file.write('\n]')
    
    # Close the files for the day
    data_file.close()
    log_file.close()

""" ****************************************************************************
                            HTTP REQUEST METHODS
**************************************************************************** """
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

def logExceptionAndWait(log_file, error_message, wait_time=5):
    '''
    Logs an exception error mesage and waits to return.
    
    param: log_file,      the file at which to log an error message
    param: error_message, the message to log for this error
    param: wait_time,     the time to wait (in seconds) before returning
    '''
    error_time = time.localtime()
    error_minute = error_time.tm_min
    error_hour = error_time.tm_hour
    if error_minute < 10:
        error_min_str = '0' + str(error_minute)
    else:
        error_min_str = str(error_minute)
    
    log_file.write('Error at ' + str(error_hour) + ':' + error_min_str
                   + ': ' + error_message + '\n')
    log_file.flush()
    
    time.sleep(wait_time)

""" ****************************************************************************
                            DATA PROCESSING METHODS
**************************************************************************** """

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

def processSong(song):
    '''
    Processes a song, extracting information about it into a dictionary.
    
    Does not include information that requires knowledge about other songs, such
    as whether or not it tied with the winner, or was the song played. 
    
    param: song,  a song object (in standard rainwave format)
    return: a dictionary of information about the song
    '''
    was_requested, requester = requestInfo(song)
    song_info = { 'id' : song['id'],
                  'title' : song['title'],
                  'artists' : pullArtists(song),
                  # rainwave used to allow for more than one album.
                  'album_name' : song['albums'][0]['name'],
                  'album_id' : song['albums'][0]['id'],
                  'album_average_rating' : song['albums'][0]['rating'],
                  'groups' : pullGroups(song),
                  'votes' : song['entry_votes'],
                  'requested' : was_requested,
                  'requester' : requester,
                  'origin_station' : song['origin_sid'],
                  'average_rating' : song['rating'],
                  'length' : song['length'] }
    
    return song_info

def pullGroups(song):
    '''
    Gets information about the groups a song belongs to.
    
    param: song,  the song (in standard rainwave format) to get group info from
    return: an array of dictionaries containing information about the groups.
    '''
    groups = []
    for id_object in song['groups']:
        group_dict = { 'name' : id_object['name'],
                       'id' : id_object['id'] }
        groups.append(group_dict)
        
    return groups

def pullArtists(song):
    '''
    Gets information about the artists who contributed to a song.
    
    param: song,  the song (in standard rainwave format) to get artist info from
    return: an array of dictionaries containing information about the artists.
    '''
    artists = []
    for id_object in song['artists']:
        artist_dict = { 'name' : id_object['name'],
                        'id' : id_object['id'] }
        artists.append(artist_dict)
        
    return artists

def parseTime(epoch_time):
    '''
    Turns the epoch time into a dictionary of information.
    
    Format (order may vary):
    {
        'day' : value 0-6 (Sunday - Sat)
        'half_hour_block' : value 0-47 (0:00 - 23:30)
    }
    
    param: epoch_time, the time in seconds from epoch
    return:            low-res infomation about the time, as described above
    '''
    time_split = time.localtime(epoch_time)
    day = (time_split.tm_wday + 1)%7 #Sunday as the first day of the week
    half_hour = time_split.tm_hour*2 + (0 if time_split.tm_min < 29 else 1)
    
    parsed_time = { 'day' : day,
                    'half_hour_block' : half_hour }
    return parsed_time

main()
