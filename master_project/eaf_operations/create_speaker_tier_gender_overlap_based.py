import xml.etree.ElementTree as ET
import os
import copy

def add_speaker_tier(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    # create speaker tier
    speaker_tier = ET.Element('TIER')
    # reference the vocabulary that is already defined for the "faces" tier
    speaker_tier.set('LINGUISTIC_TYPE_REF', 'ppl')
    speaker_tier.set('TIER_ID', 'speaker')

    
    # helper function to find the cve_id for a vocab
    speaker_vocabulary = root.findall('CONTROLLED_VOCABULARY/CV_ENTRY_ML')
    def find_vocab_referenceID(vocab):
        for entry in speaker_vocabulary:
            if entry.find('CVE_VALUE').text == vocab:
                return entry.attrib['CVE_ID']
        return ''

    # helper function to extract the timestamp (in milliseconds) for a specific timeslot 
    def get_timeslot_ms(timeslot_id):
        time_slot = root.find('.//TIME_SLOT[@TIME_SLOT_ID="' + timeslot_id + '"]')
        return int(time_slot.attrib['TIME_VALUE'])
    
    # checks if there is a new face annotation within 0.5s after a given timestamp
    def new_face_shortly_after(timestamp):
        faces = root.find('./TIER[@TIER_ID="faces"]')
        for annotation in faces.findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
            start_time = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF1'])
            if timestamp <= start_time <= timestamp + 500:
                return annotation.find('ANNOTATION_VALUE').text
        return None
    
    # returns the most recent annotation before the given timestamp
    def get_recent_annotation(timestamp):
        faces = root.find('./TIER[@TIER_ID="faces"]')
        for annotation in faces.findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
            start_time = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF1'])
            end_time = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF2'])
            if start_time <= timestamp <= end_time:
                return annotation.find('ANNOTATION_VALUE').text
        return None
        
    # checks if a speaker duration has an overlaps with a voiceover, which concludes that the speaker is a reporter
    def check_if_voiceover(start_time, end_time):
        voiceover_annotations = root.findall('./TIER[@TIER_ID="captions: talking"]/ANNOTATION/ALIGNABLE_ANNOTATION[ANNOTATION_VALUE="voiceover"]')
        for voiceover in voiceover_annotations:
            start_time_voiceover = get_timeslot_ms(voiceover.attrib['TIME_SLOT_REF1'])
            end_time_voiceover = get_timeslot_ms(voiceover.attrib['TIME_SLOT_REF2'])
            # calculate the overlap of the speaker duration with the voiceover duration and the other way around, so we check if they fully overlap
            #ovp_speaker = max(0, min(end_time, end_time_voiceover) - max(start_time, start_time_voiceover)) / (end_time_voiceover - start_time_voiceover)
            ovp_voiceover = max(0, min(end_time, end_time_voiceover) - max(start_time, start_time_voiceover)) / (end_time - start_time)
            """DEPRECATED COMMENT: only if speaker and voiceover duration overlap each other with a min. of 90% we can assume that the voiceover is done by a reporter,
            else it could be an anchor who started speaking earlier (e.g. talking-head of an anchor transitioned into a voiceover => anchor is still the speaker)"""
            # only if the whole speaker duration is overlapped by a voiceover with an overlap > 90%, we can assume, that the voiceover is done by a repoter
            if ovp_voiceover > 0.9: #DEPRECATED: and ovp_speaker > 0.9
                return True
        return False

    # return the face annotation which overlaps the longest during the speaker duration 
    def get_longest_overlap(start_time_speaker, end_time_speaker):
        faces = root.find('./TIER[@TIER_ID="faces"]')
        overlaps = {}
        for annotation in faces.findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
            start_time_face = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF1'])
            end_time_face = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF2'])
            # calculate overlap
            ovp = max(0, min(end_time_speaker, end_time_face) - max(start_time_speaker, start_time_face)) / (end_time_speaker - start_time_speaker)
            face_label = annotation.find('ANNOTATION_VALUE').text
            # sum up overlap percentages of the intersection between all faces with the same caption and store summed overlap percentages in dictionary.
            if face_label in overlaps:
                overlaps[face_label] += ovp
            else:
                overlaps[face_label] = ovp
        # convert dict to list of tuples for easier access
        overlaps_list = [(face_label, ovp) for face_label, ovp in overlaps.items()]
        # get longest overlap and return the face label associated with it
        longest_ovp = max(overlaps_list, key=lambda item:item[1])
        minutes, seconds = divmod(start_time_speaker / 1000, 60) # debugging only
        if longest_ovp[1] == 0:
            print("NO_FACE_OVP:", xmlfile.name, f'{minutes:0>2.0f}:{seconds:.0f}')
            return 'NO_FACE_OVERLAP'
        elif longest_ovp[1] <= 0.6: 
            print("LOW_CONFIDENCE:", f'({longest_ovp[0]}, {longest_ovp[1]:.2f})', os.path.basename(xmlfile.name), f'{minutes:0>2.0f}:{seconds:.0f}')
            return 'LOW_CONFIDENCE'
        else:
            return longest_ovp[0]
        

    # speaker-gender annotations act as a speaker diarization, so we can save roles of speakers that have already appeared before  
    known_speakers = {}
    annotation_count = 0
    # iterate over every speaker-gender annotation, TODO: Use xpath instead of for and if
    for tier in root.findall('TIER'):
        if tier.attrib['TIER_ID'] == 'speaker-gender':
            for annotation in tier.findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
                # for each situation annotation, create a respective speaker annotation in consideration of our defined rules
                # this way, there can only be 1 speaker at max for the duration of a news situation
                annotation_dupe = copy.deepcopy(annotation)
                annotation_dupe.set('ANNOTATION_ID', 'sp' + str(annotation_count))
                start_time = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF1'])
                end_time = get_timeslot_ms(annotation.attrib['TIME_SLOT_REF2'])
                # first check if we already know the speaker
                if annotation.find('ANNOTATION_VALUE').text in known_speakers:
                    speaker_role = known_speakers[annotation.find('ANNOTATION_VALUE').text]
                    annotation_dupe.find('ANNOTATION_VALUE').text = speaker_role
                    annotation_dupe.set('CVE_REF', find_vocab_referenceID(speaker_role))
                # then we check if the speaker is talking during a voiceover annotation
                elif check_if_voiceover(start_time, end_time):
                    annotation_dupe.find('ANNOTATION_VALUE').text = "reporter"
                    annotation_dupe.set('CVE_REF', find_vocab_referenceID("reporter"))
                    known_speakers[annotation.find('ANNOTATION_VALUE').text] = "reporter"
                # if none of the above match, we search the face that has the longest overlap with the speaker duration
                else:
                    face = get_longest_overlap(start_time, end_time)
                    annotation_dupe.find('ANNOTATION_VALUE').text = face
                    annotation_dupe.set('CVE_REF', find_vocab_referenceID(face))
                    if face not in ['NO_FACE_OVERLAP', 'LOW_CONFIDENCE']:
                        known_speakers[annotation.find('ANNOTATION_VALUE').text] = face

                annotation_tag = ET.Element('ANNOTATION')
                annotation_tag.append(annotation_dupe)

                speaker_tier.append(annotation_tag)    
                annotation_count += 1
            
            # index 3 is always where the xml tier elements start, so the speaker tier is always on top
            root.insert(3, speaker_tier)
            # prettify xml
            ET.indent(tree, space="\t", level=0)
            tree.write(result_dir + os.path.basename(xmlfile.name).split(".")[0] + "-SPEAKER.eaf", encoding="utf-8", xml_declaration=True)
            return annotation_count
        
    return 0


"""TODO: TAKE PATH FROM ARGUMENT PARSER"""
annot_dir = "C:\\Users\\Judi\\Desktop\\Data\\Annotations\\All_Default\\"
result_dir = "C:\\Users\\Judi\\Desktop\\Data\\Annotations\\Annotations_Speaker\\"
# Annotation/ (including all the annotations) and Annotations_SPEAKER/ directories have to be in same directory as this script
speaker_annotation_counter = 0
for file in os.listdir(annot_dir):
    if file.endswith(".eaf"):
        with open(annot_dir + file) as xmlfile:
            speaker_annotation_counter += add_speaker_tier(xmlfile)
print("Total speaker annotations:", speaker_annotation_counter)