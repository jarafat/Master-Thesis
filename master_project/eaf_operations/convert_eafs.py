import xml.etree.ElementTree as ET
import json
import os

def import_timelines_from_eaf(xmlfile):
    # create element tree object
    tree = ET.parse(xmlfile)

    # get root element
    root = tree.getroot()

    # get time spans
    timeslots = {}
    for timeslot in root.findall("TIME_ORDER/TIME_SLOT"):
        timeslots[timeslot.attrib["TIME_SLOT_ID"]] = timeslot.attrib

    # logging.debug(timeslots)

    timelines = []
    # EXTENSION BY JUDI: include video filename in the resulting .json
    filename = root.find("./HEADER/MEDIA_DESCRIPTOR[@MEDIA_URL]").get("MEDIA_URL")
    timelines.append({"video_fn": os.path.basename(filename)})

    # findall timelines
    annotations = 0
    for timeline in root.findall("TIER"):
        timeline_segments = []

        for annotation in timeline.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            start_time = timeslots[annotation.attrib["TIME_SLOT_REF1"]]["TIME_VALUE"]
            end_time = timeslots[annotation.attrib["TIME_SLOT_REF2"]]["TIME_VALUE"]

            for annotations_label in annotation:
                timeline_segments.append(
                    {"start": int(start_time) / 1000, "end": int(end_time) / 1000, "label": annotations_label.text}
                )
                annotations += 1
        timelines.append({"name": timeline.attrib["TIER_ID"], "segments": timeline_segments})

    # logging.debug(timelines)
    # logging.info(f"{len(timelines)} timelines with {annotations} annotations found!")
    return timelines


"""TODO: TAKE PATH FROM ARGUMENT PARSER"""
annot_dir = "C:\\Users\\Judi\\Desktop\\Data\\Annotations\\New_Annots\\"
result_dir = "C:\\Users\\Judi\\Desktop\\Data\\Annotations\\New_Annots_JSON\\"
for xfile in os.listdir(annot_dir):
    if xfile.endswith(".eaf"):
        annot_dict = {}
        timelines = import_timelines_from_eaf(annot_dir + xfile)

        annot_dict["video_fn"] = timelines[0]["video_fn"]

        for annot in timelines[1:]:
            if annot["name"] == 'Speaker':
                annot["name"] = 'speaker'
            annot_dict[annot["name"]] = annot

        json.dump(annot_dict, open(result_dir + xfile.replace(".eaf", ".json"), "w"))