import cv2
import os
import textwrap



def get_character_images_move_array(character_name,folder_path):
    character_games_images_array = []
    files_in_directory = os.listdir(f"character/{character_name}/{folder_path}") 
    for images in files_in_directory:
        full_img_path = f"character/{character_name}/{folder_path}/{images}"
        character_games_images_array.append(full_img_path)

    return character_games_images_array


width_btn_select = 244
height_btn_select = 190
width_full_img = 1280
height_full_img = 720
bg_window_name = "narut_game"
stage_full_path = "dataset/intro/title.png"
x_cur_roi = 30
y_cur_roi = 480
w_cur_roi = 110 + x_cur_roi
h_cur_roi = 110 + y_cur_roi
speed = 10
character_index_right = 0
character_index_dash = 0
character_index_move_one = 0
character_index_move_two = 0
character_index_move_three = 0
flip_character = False


start_game = False
char_select = False
stage_select = False
land_page = True
home_page = True
about_page = False

game_color = (193, 145, 1)
game_text_color = (255,255,255)


play_button = {"text": "play","x":500,"y":350,"w":800,"h":450}
home_button = {"text":"Home","x":1170,"y":10,"w":1270,"h":70}
about_button = {"text": "controls", "x" : 500,"y": 500,"w": 800,"h": 600}

controls_info = [
    {"text":"Controls","x":30,"y":50,"w":520,"h":110},
    {"text":"A D : Move right / Move Left" ,"x":30,"y":150,"w":520,"h":210},
    {"text":"S : Dash" ,"x" :30,"y":250,"w":520,"h":310},
    {"text":"j : Combo One" ,"x" :30,"y":350,"w":520,"h":410},
    {"text":"k : Combo Two" ,"x" :30,"y":450,"w":520,"h":510},
    {"text":"l : Combo Three" ,"x" :30,"y":550,"w":520,"h":610},
    {"text":"q : Exit" ,"x" :30,"y":650,"w":520,"h":710},
]



characters = [
    {"name": "deidara", "x": 10, "y": 95, "w": width_btn_select, "h": height_btn_select,"choose":"character/deidara/logo/logo.jpg"},
    {"name": "pain", "x": 10, "y": 290, "w": width_btn_select, "h": height_btn_select,"choose":"character/pain/logo/logo.jpg"},
    {"name": "itachi", "x":10, "y": 485, "w": width_btn_select, "h": height_btn_select,"choose":"character/itachi/logo/logo.jpg"},
    {"name": "kakashi", "x": 264, "y": 95, "w": width_btn_select, "h": height_btn_select,"choose":"character/kakashi/logo/logo.jpg"},
    {"name": "shikamaru", "x": 264, "y": 290, "w": width_btn_select, "h": height_btn_select,"choose":"character/shikamaru/logo/logo.jpg"},
    {"name": "suigetsu", "x": 264, "y": 485, "w": width_btn_select, "h": height_btn_select,"choose":"character/suigetsu/logo/logo.jpg"},
    {"name": "naruto", "x": 518, "y": 95, "w": width_btn_select, "h": height_btn_select,"choose":"character/naruto/logo/logo.jpg"},
    {"name": "jiraya", "x": 518, "y": 290, "w": width_btn_select, "h": height_btn_select,"choose":"character/jiraya/logo/logo.jpg"},
    {"name": "karin", "x": 518, "y": 485, "w": width_btn_select, "h": height_btn_select,"choose":"character/karin/logo/logo.jpg"},
    {"name": "sai", "x": 772, "y": 95, "w": width_btn_select, "h": height_btn_select,"choose":"character/sai/logo/logo.jpg"},
    {"name": "fukasaku", "x": 772, "y": 290, "w": width_btn_select, "h": height_btn_select,"choose":"character/fukasaku/logo/logo.jpg"},
    {"name": "jugo", "x": 772, "y": 485, "w": width_btn_select, "h": height_btn_select,"choose":"character/jugo/logo/logo.jpg"},
    {"name": "sakura", "x": 1026, "y": 95, "w": width_btn_select, "h": height_btn_select,"choose":"character/sakura/logo/logo.jpg"},
    {"name": "orochimaru", "x": 1026, "y": 290, "w": width_btn_select, "h": height_btn_select,"choose":"character/orochimaru/logo/logo.jpg"},
    {"name": "sasuke", "x": 1026, "y": 485, "w": width_btn_select, "h": height_btn_select,"choose":"character/sasuke/logo/logo.jpg"},
]


stages = [
    {"name": "base", "x": 10, "y": 95, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/base.png"},
    {"name": "base_2", "x": 10, "y": 290, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/base_2.png"},
    {"name": "base_3", "x": 10, "y": 485, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/base_3.png"},
    {"name": "snake", "x": 264, "y": 95, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/snake.png"},
    {"name": "lab", "x": 264, "y": 290, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/lab.png"},
    {"name": "montain", "x": 264, "y": 485, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/mon.png"},
    {"name": "four_tails", "x": 518, "y": 95, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/four_tails.png"},
    {"name": "pain forest", "x": 518, "y": 290, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/four_tails_two.png"},
    {"name": "konoha", "x": 518, "y": 485, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/konoha.png"},
    {"name": "bridge", "x": 772, "y": 95, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/bridge.png"},
    {"name": "forest", "x": 772, "y": 290, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/forest.png"},
    {"name": "spider", "x": 772, "y": 485, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/spider.png"},
    {"name": "town", "x": 1026, "y": 95, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/town.png"},
    {"name": "roof", "x": 1026, "y": 290, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/roof.png"},
    {"name": "academy", "x": 1026, "y": 485, "w": width_btn_select, "h": height_btn_select, "stage_path":"dataset/stages/academy.png"},
]


def game_menu(event, x, y, flags, params):
    global character_name, playable_character, start_game,logo_path,stage_select,char_select,stage_full_path,land_page,game_color,about_page
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_game == True or about_page == True:
            if home_button["x"] < x < home_button["w"] +  home_button["x"] and  home_button["y"] <y <  home_button["h"] + home_button["y"]:
                char_select = False
                stage_select = False
                start_game = False
                about_page = False
                stage_full_path = "dataset/intro/title.png"
                land_page = True

        if land_page:
            if play_button["x"] < x < play_button["w"] and  play_button["y"] <y <  play_button["h"]:
                land_page = False
                char_select = True
                about_page = False
                stage_full_path = "dataset/intro/home.png"

            if about_button["x"] < x < about_button["w"] and  about_button["y"] <y <  about_button["h"]:
                land_page = False
                about_page = True
                stage_full_path = "dataset/intro/about.jpg"

        elif char_select:
            for char_info in characters:
                if char_info["x"] < x < char_info["w"] + char_info["x"] and char_info["y"] < y < char_info["h"] + char_info["y"]:
                    character_name = char_info["name"]
                    playable_character = f"character/{character_name}/move_right/0.jpg"
                    logo_path = f"character/{character_name}/logo/logo.jpg"
                    char_select = False
                    about_page = False
                    stage_select = True
        
        elif stage_select:
            for stage in stages:
                if stage["x"] < x < stage["w"] + stage["x"] and stage["y"] < y < stage["h"] + stage["y"]:
                    stage_select = False
                    about_page = False
                    start_game = True
                    stage_full_path = stage["stage_path"]
            
  

cv2.namedWindow(bg_window_name)
cv2.setMouseCallback(bg_window_name,game_menu)


while True:
    background_img = cv2.imread(stage_full_path)
    background_img = cv2.resize(background_img, (width_full_img, height_full_img), cv2.INTER_LINEAR)


    if land_page:
        cv2.rectangle(background_img, (play_button["x"], play_button["y"]),(play_button["w"], play_button["h"]), game_color, -1)
        cv2.putText(background_img, play_button["text"].capitalize(), (play_button["x"] + 79, play_button["y"] + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, game_text_color, 2, cv2.LINE_AA)
        cv2.rectangle(background_img, (about_button["x"], about_button["y"]),(about_button["w"], about_button["h"]), game_color, -1)
        cv2.putText(background_img, about_button["text"].capitalize(), (about_button["x"] + 25, about_button["y"] + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, game_text_color, 2, cv2.LINE_AA)


    if about_page:
        cv2.rectangle(background_img, (home_button["x"], home_button["y"]),(home_button["w"], home_button["h"]), game_color, -1)
        cv2.putText(background_img, home_button["text"].capitalize(), (home_button["x"] + 5, home_button["y"] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)
        for info in controls_info:
            cv2.rectangle(background_img, (info["x"], info["y"]), (info["w"],info["h"]), game_color, -1)
            cv2.putText(background_img, info["text"], (info["x"] + 10, info["y"] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)


    if char_select:
        cv2.rectangle(background_img, (0,0),(width_full_img, 50), game_color, -1)
        cv2.putText(background_img, "Choose Character", (500,35), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)
        for char_info in characters:
            # cv2.rectangle(background_img, (char_info["x"], char_info["y"]), (char_info["w"] + char_info["x"], char_info["h"] + char_info["y"]), game_color, -1)
            char_img = cv2.imread(char_info["choose"])
            char_img = cv2.resize(char_img,(char_info["w"],char_info["h"]),cv2.INTER_LINEAR)
            x_char = char_info["x"]
            y_char = char_info["y"]
            w_char = x_char + char_info["w"]  
            h_char = y_char + char_info["h"]
            background_img[y_char:h_char, x_char:w_char] = char_img
            cv2.putText(background_img, char_info["name"].capitalize(), (char_info["x"] + 20, char_info["y"] + 180), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)


    if stage_select:
        cv2.rectangle(background_img, (0,0),(width_full_img, 50), game_color, -1)
        cv2.putText(background_img, "Choose Stage", (500,35), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)
        for stage in stages:
            # cv2.rectangle(background_img, (stage["x"], stage["y"]), (stage["w"] + stage["x"], stage["h"] + stage["y"]), game_color, -1)
            stg_img = cv2.imread(stage["stage_path"])
            stg_img = cv2.resize(stg_img,(stage["w"],stage["h"]),cv2.INTER_LINEAR)
            x_stg= stage["x"]
            y_stg = stage["y"]
            w_stg = x_stg + stage["w"]  
            h_stg = y_stg + stage["h"]
            background_img[y_stg:h_stg, x_stg:w_stg] = stg_img
            cv2.putText(background_img, stage["name"].capitalize(), (stage["x"] + 60, stage["y"] + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)


    if start_game:
        cv2.rectangle(background_img, (home_button["x"], home_button["y"]),(home_button["w"], home_button["h"]), game_color, -1)
        cv2.putText(background_img, home_button["text"].capitalize(), (home_button["x"] + 5, home_button["y"] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, game_text_color, 2, cv2.LINE_AA)
        logo_character = cv2.imread(logo_path)
        logo_character = cv2.resize(logo_character, (120, 150))
        logo_height, logo_width, _ = logo_character.shape
        background_img[10:10 + logo_height, 10:10 + logo_width] = logo_character



        char = cv2.imread(playable_character)
        if flip_character:
            char = cv2.flip(char,1)


        char = cv2.resize(char, (w_cur_roi - x_cur_roi, h_cur_roi - y_cur_roi), cv2.IMREAD_UNCHANGED)
        background_img[y_cur_roi:h_cur_roi, x_cur_roi:w_cur_roi] = char


    cv2.imshow(bg_window_name, background_img)
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        break



    if start_game:
        if k == ord('d') :  
            character_move_right_array = get_character_images_move_array(character_name,"move_right")
            flip_character = False
            if x_cur_roi <= 1160:
                x_cur_roi += speed  
                w_cur_roi += speed
                character_index_right = (character_index_right + 1) % len(character_move_right_array)
                playable_character = character_move_right_array[character_index_right]
        

        elif k == ord('a') : 
            character_move_left_array = get_character_images_move_array(character_name,"move_right")
            flip_character = True 
            if x_cur_roi >= 10:
                x_cur_roi -= speed  
                w_cur_roi -= speed 
                character_index_right = (character_index_right + 1) % len(character_move_left_array)
                playable_character = character_move_left_array[character_index_right]


        elif k == ord('s') :  
            character_move_dash_array = get_character_images_move_array(character_name,"dash")
            if flip_character and x_cur_roi > 0:
                x_cur_roi -= speed  
                w_cur_roi -= speed
            elif x_cur_roi < 1170:
                x_cur_roi += speed  
                w_cur_roi += speed
            character_index_dash = (character_index_dash + 1) % len(character_move_dash_array)
            playable_character = character_move_dash_array[character_index_dash]
        

        elif k == ord('j') :  
            character_move_one_array = get_character_images_move_array(character_name,"move_one")
            character_index_move_one = (character_index_move_one + 1) % len(character_move_one_array)
            playable_character = character_move_one_array[character_index_move_one]
        
        elif k == ord('k') :  
            character_move_two_array = get_character_images_move_array(character_name,"move_two")
            character_index_move_two = (character_index_move_two + 1) % len(character_move_two_array)
            playable_character = character_move_two_array[character_index_move_two]
        
        elif k == ord('l') :  
            character_move_three_array = get_character_images_move_array(character_name,"move_three")
            character_index_move_three = (character_index_move_three + 1) % len(character_move_three_array)
            playable_character = character_move_three_array[character_index_move_three]