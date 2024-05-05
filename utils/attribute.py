import re
import os 
import pandas as pd 
from textblob import TextBlob
#caption = TextBlob(caption).correct() 


def isWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def get_substring(caption, search_word):
    cap_words = caption.split(" ")
    if not search_word in cap_words:
        return 0
    
    else:
        pos = cap_words.index(search_word)
        start_index = pos
        end_index = pos
        count = 0
        while not start_index == 0:
            start_index -= 1
            count += 1
            
            if count == 2: break 

        count = 0
        while not end_index == len(cap_words) - 1:
            end_index += 1
            count += 1
            if count == 2: break 

        return " ".join(cap_words[start_index : end_index+1])


caps_all_attributes = ["wearing earrings", "wearing hat", "wearing lipstick", "wearing necklace", "wearing necktie", 
                       "heavy makeup", "receding hairline", "bald", "bangs", "straight hair", 
                       "wavy hair",  "gray hair", "blond hair", "black hair", "brown hair",
                       "bags under eyes", "arched eyebrows", "eyeglasses", "narrow eyes", "bushy eyebrows",
                       "goatee", "mustache",  "no beard", "male", "sideburns",  
                       "5 o'clock shadow", "attractive", "young", "smiling", "blurry", 
                       "chubby", "double chin",  "high cheekbones", "rosy cheeks", "mouth slightly open",  
                       "oval face", "pale skin", "pointy nose", "big lips", "big nose"]


def get_attr_vector(caption):
    if not isinstance(caption, str):
        return [0.0] * 40
    
    else: 
        caption = caption.lower()
        caption = " ".join(caption.split())
        caption = re.sub('[^\w\s]', "", caption)

        ########## wearing & makeup
        str_wearing = ["wearing earrings", "wearing hat", "wearing lipstick", "wearing necklace", "wearing necktie", "heavy makeup"] #6
        ls_wearing = [0.0] * len(str_wearing)

        #exact match
        for i, item in enumerate(str_wearing):
            if item in caption:
                ls_wearing[i] = 1.0

            elif item.split(" ")[-1] in caption:
                ls_wearing[i] = 1.0

        if isWord("makeup")(caption):
            ls_wearing[2] = 1.0

        #print(ls_wearing)


        ################# hair & head
        str_hair = ["receding hairline", "bald", "bangs", "straight hair", "wavy hair", "gray hair", "blond hair", "black hair", "brown hair"] #9
        ls_hair =  [0.0] * len(str_hair)
        for i, item in enumerate(str_hair):
            if item in caption:
                ls_hair[i] = 1.0


        if "receding" in caption:
                ls_hair[0] = 1.0
                ls_hair[1] = 0.0

        if "wavy" in caption:
                ls_hair[3] = 0.0
                ls_hair[4] = 1.0

        elif "straight" in caption:
                ls_hair[3] = 1.0
                ls_hair[4] = 0.0

        if "blond" in caption:
                ls_hair[5] = 0.0
                ls_hair[6] = 1.0
                ls_hair[7] = 0.0
                ls_hair[8] = 0.0

        if isWord("hair")(caption):
            if "gray" in caption:
                ls_hair[5] = 1.0
                ls_hair[7] = 0.0
                ls_hair[8] = 0.0
                
            elif "black" in caption:
                ls_hair[5] = 0.0
                ls_hair[7] = 1.0
                ls_hair[8] = 0.0

            elif "brown" in caption:
                ls_hair[5] = 0.0
                ls_hair[7] = 0.0
                ls_hair[8] = 1.0
        #print(ls_hair)

        #################### eye
        str_eye = ["bags under eyes", "arched eyebrows", "eyeglasses", "narrow eyes", "bushy eyebrows"] #5
        ls_eye =  [0.0] * len(str_eye)
        for i, item in enumerate(str_eye):
            if item in caption:
                ls_eye[i] = 1.0

            elif item.split(" ")[0] in caption:
                ls_eye[i] = 1.0

        if "eyeglass" in caption:
                ls_eye[2] = 1.0

        if "thin eyebrows" in caption:
                ls_eye[4] = -1.0


        ################# man
        str_man = ["goatee", "mustache",  "no beard", "male", "sideburns",  "5 o'clock shadow"] #6
        ls_man =  [0.0] * len(str_man)
        for i, item in enumerate(str_man):
            if item in caption:
                ls_man[i] = 1.0

        if "moustache" in caption:
                ls_man[1] = 1.0
                ls_man[3] = 1.0

        if "sideburn" in caption:
                ls_man[4] = 1.0
                ls_man[3] = 1.0

        if ls_man[2] != 1.0 and isWord("beard")(caption):
            ls_man[2] = -1.0 
            
        for sub in ["man", "he", "He"]:
            if sub in caption: ls_man[3] = 1.0

        for sub in ["woman", "she", "She"]:
            if sub in caption: ls_man[3] = -1.0

        if ls_man[0] == 1.0 or ls_man[1] == 1.0:
            ls_man[3] = 1.0

        if ls_man[3] == -1.0: #for female
            ls_man[0] = 0.0
            ls_man[1] = 0.0
            ls_man[2] = 1.0
            ls_man[4] = 0.0
            ls_man[5] = 0.0
        #print(ls_man)

        ######## mostly female face
        str_female = ["attractive", "young", "smiling", "blurry", "chubby", "double chin",  "high cheekbones", "rosy cheeks"] #8
        ls_female =  [0.0] * len(str_female)

        for i, item in enumerate(str_female):
            if item in caption:
                ls_female[i] = 1.0

            elif item.split(" ")[0] in caption:
                ls_female[i] = 1.0

        if "cheekbone" in caption:
                ls_female[6] = 1.0

        #print(ls_female)


        str_face = "mouth slightly open",  "oval face", "pale skin", "pointy nose", "big lips", "big nose" #6
        ls_face =  [0.0] * len(str_face)

        for i, item in enumerate(str_face):
            if item in caption:
                ls_face[i] = 1.0

            elif (i <= 3) and (item.split(" ")[0] in caption):
                ls_face[i] = 1.0


        if ls_face[2] == 0.0 :
            if "fair" in caption:
                ls_face[2] = -1.0

        if "pointed" in caption:
            ls_face[3] = 1.0


        if "lips" in caption:
            if "full" in caption:
                ls_face[4] = 1.0
            elif "small" in caption:
                ls_face[4] = -1.0

        if "nose" in caption:
            if "wide" in caption:
                ls_face[5] = 1.0
            elif "small" in caption:
                ls_face[5] = -1.0

        #print(ls_face)
        return ls_wearing + ls_hair + ls_eye + ls_man + ls_female + ls_face 



if __name__ == "__main__":
    with open("./data/face2text/annotations/output.csv", "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
        cnt = 0
        all_list = []
        for i, cap in enumerate(captions[2:3]):
            cap = cap.replace("\ufffd\ufffd", " ")
            #image_file = cap.split(",")[0]
            cap_int =  get_attr_vector("".join(cap.split(",")[1:])) #[int(c) for c in cap.split(",")[1:]]
            """
            if cap_int in all_list:
                print("############################## match")
                print(cap)
                print("found match in ", i)
                all_list.append(cap_int)
                print("here , ", all_list.index(cap_int))
            else:
                 all_list.append(cap_int)
            """

            print(cap)
            print(cap_int)