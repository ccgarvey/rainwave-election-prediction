"""
This program is used to parse the data recorded in JSON format into simpler
data, in which each song's information is on one line.
"""

import sys #command-line arguments
import json #to allow reading of data
import os #to check if the output file exists already or not


def write_value(out_file, value, isLast=False):
    """
    Writes a value to the output file. Ends with ", " unless isLast is True.
    
    param: out_file, the output file 
    param: value,    the value to write.
    param: isLast,   indicates whether the value is the last in the line of text
    """
    
    out_file.write(str(value) + ('\n' if isLast else ', '))
    
def main():
    # Check and parse command line arguments
    if len(sys.argv) != 3:
        print('Usage: python parse_data_restricted <file_to_parse> '
              + '<output_file>')
        return
    
    json_filename = sys.argv[1]
    out_filename = sys.argv[2]
    
    # Load the file to be parsed
    json_data_file = open(json_filename, 'r')
    json_data = json.loads(json_data_file.read())
    
    # Open the output file, with appending enabled.
    outfile_new = not os.path.exists(out_filename)
    output_file = open(out_filename, 'a')
    
    # Write the first line of the output file if it's new
    if outfile_new:
        output_file.write('song_ID, number_songs, prev_5_elec_vote_avg, '
                        + 'prev_elec_votes, prev_winner_rating, day_of_week, '
                        + 'half_hour_block, song_avg_rating, album_avg_rating, '
                        + 'length_sec, requested, winner\n')
    
    
    for election in json_data:
        # If the winner was chosen randomly, skip the election.
        # NOTE: this is a dicey choice and may not make sense. Kills ~half.
        if election['chosen_randomly']:
            continue
        
        # Write out song info as specified above.
        for song in election['songs']:
            write_value(output_file, song['id'])
            write_value(output_file, election['number_songs'])
            write_value(output_file, election['prev_5_elec_vote_avg'])
            write_value(output_file, election['prev_elec_votes'])
            write_value(output_file, election['prev_winner_rating'])
            write_value(output_file, election['time']['day'])
            write_value(output_file, election['time']['half_hour_block'])
            write_value(output_file, song['average_rating'])
            write_value(output_file, song['album_average_rating'])
            write_value(output_file, song['length'])
            write_value(output_file, '1' if song['requested'] else '0')
            write_value(output_file, '1' if song['was_played'] else '0', True)
    
    # Close the input and output files.
    output_file.close()
    json_data_file.close()

main()
