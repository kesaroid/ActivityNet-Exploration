'''
ffmpeg -i <filename> -vf "select=eq(pict_type\,I)" -vsync vfr -start_number <start_time> <out_path>/%d.jpg -hide_banner -loglevel panic
'''

import time
import json
import random
import cv2
import sys
import pandas as pd

yolo_path = 'C:/Users/Kesar/Documents/Artificial intelligence/YOLO/pytorch-YOLOv4/'
sys.path.append(yolo_path)

from tool.utils import *
from tool.darknet2pytorch import Darknet

def prepare_yolo():
    cfgfile = yolo_path + 'cfg/yolov4.cfg'
    weightfile = yolo_path + 'cfg/yolov4.weights'
    
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    m.cuda()
    
    return m, cfgfile, weightfile

def detect_cv2(img, m, cfgfile, weightfile):
    namesfile = yolo_path + 'data/coco.names'
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda = 1)

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, class_names=class_names)

def __draw_label(img, text, pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    color = (255, 255, 255)
    thickness = cv2.FILLED
    margin = 1

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), (0, 0, 0), thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def play_video(url, duration, captions, cfgfile, weightsfile):
    import pafy
    import vlc

    video = pafy.new(url)
    best = video.getbest()

    # Instance = vlc.Instance()
    # player = Instance.media_player_new()
    # Media = Instance.media_new(best.url)
    # Media.get_mrl()
    # Media.add_option('start-time=' + str(round(duration[0])))
    # player.set_media(Media)

    # if run_yolo:
    #     best.download(filepath='temp.mp4')

    # player.play()
    # time.sleep(duration[1] - duration[0])
    # player.stop()

    cap = cv2.VideoCapture(best.url)
    start = int(duration[0]); end = int(duration[1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameth = 0

    while (int(frameth/fps) < end):
        _, frame = cap.read()
        
        frameth = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        if (int(frameth/fps) >= start):
            if captions:
                for caps in captions['annotations']:
                    if (int(frameth/fps) > int(caps['segment'][0])) and (int(frameth/fps) < int(caps['segment'][1])):
                        __draw_label(frame, caps['sentence'], (5, 50))

            if run_yolo:
                detect_cv2(frame, m, cfgfile, weightfile)
            else:
                cv2.imshow(video.title, frame)

        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break    

    cap.release()
    cv2.destroyAllWindows()

def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

def stats(database, ids, show=False):

    df = pd.DataFrame(columns=('ID', 'Subset', 'Resolution', 'duration', 'url'))
    good_res = ['1280x720', '1920x1080', '720x1280']
    durations = []

    for chosen in ids:
        annotation = database.get(chosen)
        if annotation['resolution'] in good_res:
            video_time = annotation['annotations'][0]['segment'][1] - annotation['annotations'][0]['segment'][0]
            durations.append(video_time)
            row = pd.Series({'ID': chosen, 
                            'Subset': annotation['subset'], 
                            'Resolution': annotation['resolution'], 
                            'duration': video_time, 
                            'url': annotation['url']})
            df = df.append(row, ignore_index=True)

    print('Minimum duration: {:.2f}\tMaximum duration: {:.2f}\tAverage duration: {:.2f}'.format(min(durations), 
                                                                                                max(durations), 
                                                                                                (sum(durations)/len(durations))))
    if show:
        print(df)
        print(df.size)
    return df
        

def anet_captions(database, captions): 
    df = pd.DataFrame(columns=('ID', 'label', 'subset', 'resolution', 'duration', 'caption', 'url'))
    good_res = ['1280x720', '1920x1080', '720x1280']
    
    for id, annotation in captions.items():
        if len(annotation['annotations']) >= 5:
            activity = database.get(id)
            if activity['resolution'] in good_res:
                row = pd.Series({'ID': id,
                                'label': activity['annotations'][0]['label'],
                                'subset': annotation['subset'],
                                'resolution': activity['resolution'],
                                'duration': annotation['duration'],
                                'caption': [(value['segment'], value['sentence']) for value in annotation['annotations']],
                                'url': activity['url']
                })
                df = df.append(row, ignore_index=True)
    
    df.to_csv('anet_captions.csv')
    print(df)    

def explorer(database, questions, captions, label, cfgfile, weightfile, play=False):
    ids = []
    getValues = lambda key,inputData: [subVal[key] for subVal in inputData if key in subVal]
    
    for db in database.items():
        if label in getValues('label', db[1]['annotations']):
            ids.append(db[0])
    
    df = stats(database, ids, True)
    chosen = random.choice(df['ID'])
    example = database.get(chosen)

    print('Randomly chosen ID:\n', df.loc[df['ID'] == chosen].to_string(index=False, header=False))
    print('Activity Duration: {:.2f} seconds'.format(example['annotations'][0]['segment'][1] - example['annotations'][0]['segment'][0]))

    if play:
        play_video(example['url'], example['annotations'][0]['segment'], captions['database'][chosen], cfgfile, weightfile)
    
    count = 0
    for q in questions:
        if q['video_name'] == chosen:
            print('Question {}:\t{}'.format(count, q['question']))
            count += 1

def create_questions(path, question_path):
    df = pd.read_csv(path)           
    try:
        question_df = pd.read_csv(question_path)
    except:    
        question_df = pd.DataFrame(columns = ['ID', 'Questions'])

    for _, segments in df.iterrows():
        if segments['ID'] not in question_df['ID']:
            print(segments['ID'], ':\n', segments['caption'])
            question = input('Type your question here:')
            row = pd.Series({'ID': segments['ID'], 'Questions': question})
            question_df = question_df.append(row, ignore_index=True)
        
            question_df.to_csv(question_path)


def taxonomy_graph(taxonomy):
    from anytree import Node, RenderTree
    from anytree.exporter import DictExporter

    Root = Node("ActivityNet")   
    levlist_1 = []; levlist_2 = []; levlist_3 = []

    for i in range(len(taxonomy)):
        node = taxonomy[i]
        if node['parentName'] == 'Root':
            level_1 = Node(node['nodeName'], parent=Root)
            levlist_1.append(node['nodeName'])
            for j in range(len(taxonomy)):
                node_2 = taxonomy[j]
                if node_2['parentName'] in levlist_1:
                    level_2 = Node(node_2['nodeName'], parent=level_1)
                    levlist_2.append(node_2['nodeName'])
                for k in range(len(taxonomy)):
                    node_3 = taxonomy[k]
                    if node_3['parentName'] in levlist_2:
                        level_3 = Node(node_3['nodeName'], parent=level_2)
                        levlist_3.append(node_3['nodeName'])
                    for l in range(len(taxonomy)):
                        node_4 = taxonomy[l]
                        if node_4['parentName'] in levlist_3:
                            level_4 = Node(node_4['nodeName'], parent=level_3)
                    levlist_3.clear()
                levlist_2.clear()
            levlist_1.clear()

    for pre, fill, node in RenderTree(Root):
        print("%s%s" % (pre, node.name))
    
    exporter = DictExporter()
    return exporter.export(Root)
    

if __name__ == "__main__":
    activity_path = 'activitynet-qa/dataset/activitynet.json'
    qfile_path = 'activitynet-qa/dataset/train_q.json'
    caps_path = 'activitynet-captions/anet_annotations_trainval.json'
    run_yolo = False

    captions = json.load(open(caps_path, 'r'))
    questions = json.load(open(qfile_path, 'r'))
    activitynet = json.load(open(activity_path, 'r'))

    # play_video('https://www.youtube.com/watch?v=VkRjs03YEjE', None, [0.4648788119555858, 13.614300898547231])
    # taxonomy = taxonomy_graph(activitynet['taxonomy'])

    '''
    # Explorer

    if run_yolo:
        m, cfgfile, weightfile = prepare_yolo()
    else:
        m = None; cfgfile = None; weightfile = None
    
    explorer(activitynet['database'], 
            questions,
            captions, 
            "Swinging at the playground", 
            cfgfile, weightfile,
            play=True)
    '''
    # Create csv of dataset
    # anet_captions(activitynet['database'], captions['database'])

    # Create Questions
    create_questions('anet_captions.csv', 'anet_question.csv')