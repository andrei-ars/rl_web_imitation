# based on gridworld.py and tictactoe.py

import datetime
import os
import sys
import logging
from collections import deque

import numpy as np
import torch
from gym.utils import seeding
#from webdriverwrapper import Chrome
#from selenium import webdriver

NUMBER_ACTIONS = 7
MAX_STEPS = 100

WIN_REWARD, POS_REWARD, NEG_REWARD = 100, 1, 0 # 100, 1, 0

"""
class Selenium_webdriver:
    def __init__(self, url_address, driver_type="Chrome"):
        # Return chrome webdriver and open the initial webpage.
        
        from selenium import webdriver
        from bs4 import BeautifulSoup
        options = webdriver.ChromeOptions()
        options.add_argument('--ignore-certificate-errors')
        options.add_argument("--test-type")
        #options.add_experimental_option("excludeSwitches", ["enable-automation"])
        #options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(chrome_options=options)
        #from webdriverwrapper import Chrome
        #self.driver = Chrome(options=options)
        # Open a website
        #window_before = self.driver.window_handles[0]
        self.driver.get(url_address)

    def get_page_elements(self):
        
        html_soup = BeautifulSoup(html_code, 'lxml')
        elems = {}
        elems['input'] = html_soup.find_all('input')
        elems['button'] = html_soup.find_all('button')
        elements = {'input': [], 'button': []}

        for i, elem in enumerate(elems['input']):
            element = {
                'id': i, # 0, 1, 2, ...
                'name': str(elem.attrs.get('name')), # like 'email'
                'type': str(elem.attrs.get('type')), # like 'email'
                'text': str(elem.attrs.get('placeholder')), # like 'email'
                'selenium_element_type': str(elem.name),  # like 'input'
                'xpath': xpath_soup(elem),
                }
            elements['input'].append(element)
            print("input_element: {}".format(element))

        for i, elem in enumerate(elems['button']):
            element = {
                'id': i,  # 0, 1, 2, ...
                'name': str(elem.attrs.get('class')), # like ['btn', 'btn-primary']
                'type': str(elem.attrs.get('type')), # like 'submit'
                'text': str(elem.attrs.get('placeholder')), # None
                'selenium_element_type': str(elem.name),  # like 'button'
                'xpath': xpath_soup(elem),
                }
            elements['input'].append(element)
            print("input_element: {}".format(element))

        return elements

    def click(self, current_element):
        #self.driver.find_element_by_xpath()
        element = self.driver.find_element_by_name(current_element)
        # the function find_element_by_name should be implemented
        element.click()

    def enter(self, current_element, data):
        element = self.driver.find_element_by_name(current_element)
        enter_field = self.driver.find_element_by_xpath("//input[@name='{}']".format(element))
        enter_field.clear()
        data = generate_data()
        enter_field.send_keys(data)
"""

class Webdriver_imitation:

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_action = None
        self.site_elements = {
            'clickables': [0],
            'selectables': [], 
            'enterables': [0, 0]
            }

    def get_site_elements(self):
        return self.site_elements

    def is_target_achieved(self):
        #observation = self.get_observation()
        #sum_obs = np.sum(observation[:self.env_size[0]*self.env_size[1]])
        #return True if sum_obs < 0.01 else False # if all_active_elements_have_been_clicked
        if self.last_action == "CLICK"\
                and self.site_elements['enterables'][0] == 1\
                and self.site_elements['enterables'][1] == 1\
                and self.site_elements['clickables'][0] > 0:
            return True
        else:
            return False

    def action_on_element(self, action, element_number, data=None):
        #print("{} {}".format(action, element_number))
        self.last_action = action
        action_to_element_type = {"CLICK": "clickables", "ENTER": "enterables"}
        element_type = action_to_element_type[action]
        if element_number < len(self.site_elements[element_type]):
            self.site_elements[element_type][element_number] += 1
            return True
        else:
            return False
    """
    def click(self, current_element_name):
        #print("clickables:", self.site_elements['clickables'])
        #print("current_element_name:", current_element_name)
        if current_element_name in self.site_elements['clickables']:
            index = self.site_elements['clickables'].index(current_element_name)
            self.site_elements['clickables'][index] = None
            return True
        else:
            logging.warning("{} is not in the clickables list".format(current_element_name))
            return False

    def enter(self, current_element_name, data):
        if current_element_name in self.site_elements['enterables']:
            index = self.site_elements['enterables'].index(current_element_name)
            self.site_elements['enterables'][index] = None
            return True
        else:
            logging.warning("{} is not in the enterables list".format(current_element_name))
            return False
    """

def negative_reward():
    #print("-1")
    return NEG_REWARD

def positive_reward():
    #print("+1")
    return POS_REWARD


class TestDriverEnviroment:
    """This is enviroment for a test driver
    """
    def __init__(self):

        self.env_size = (3, 5)

        self.step_count = 0
        self.total_steps = 0
        self.last_num_steps = 0
        self.history_steps = deque()
        self.seed()
        #self.board = numpy.zeros((3, 3)).astype(int)
        # Prepare the webdriver
        #self.driver = Selenium_webdriver(url_address="https://google.com")
        self.driver = Webdriver_imitation()

        # Init the state
        self.chosen_type = None     # 'clickables', 'selectables', 'enterables'
        self.chosen_number = None
        self.reset()

        self.possible_actions = [
            #"CLICK-N": "Click the N-th clickable element",
            #"ENTER-RND": "Enter random text",
            #"OPEN":     "Open the given website",
            (0, "NEXT",     "Go to the next active element"),
            (1, "CLICK",    "Click on the current element"),
            (2, "CHOOSE_FIRST_CLICK",  "Choose the firse clickable element"),
            (3, "ENTER",    "Enter DATA in the current element"),
            (4, "CHOOSE_FIRST_ENTER",  "Choose the firse enterable element"),
            (5, "SELECT",    "Select the current element"),
            (6, "CHOOSE_FIRST_SELECT", "Choose the firse selectable element"),
            #(7, "HIT",      "Hit the current element"),
            #(8, "VERIFY",   "Verify the current URL"),
            #(9, "CLOSE",    "Close the current page"),
            #(10, "WAIT",     "Wait 1 sec"),
        ]
        self.possible_actions = self.possible_actions[:NUMBER_ACTIONS]
        self.cmd_to_element_type = {
            "CHOOSE_FIRST_CLICK": "clickables",
            "CHOOSE_FIRST_SELECT": "selectables",
            "CHOOSE_FIRST_ENTER": "enterables",
            "CLICK": "clickables",
            "SELECT": "selectables",
            "ENTER": "enterables"
        }
        self.action_number_to_cmd = {i: x[1] for i, x in enumerate(self.possible_actions)}
        self.action_number_to_description = {i: x[2] for i, x in enumerate(self.possible_actions)}
        self.wins = 0
        self.losses = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #def to_play(self):
    #    return 0 if self.player == 1 else 1

    def reset(self):
        #self.board = numpy.zeros((3, 3)).astype(int)
        #self.state = [0]
        self.last_num_steps = self.step_count
        self.step_count = 0
        self.driver.reset()
        self.prv_state = None
        return self.get_observation()

    def step(self, action):
        """
        action (int or str)
        """
        self.step_count += 1
        self.total_steps += 1

        # Call the webdriver and perform the action
        if type(action) is str:
            cmd = action
        else:
            action = int(action)
            cmd = self.action_number_to_cmd[action]
            #print("action type:", type(action))
            #raise Exception("Wrong the action type")

        #print("{}: wins={} (in {}), obs={} cmd={}".format(
        #    self.step_count, self.wins, self.last_num_steps, self.get_observation()[:15], cmd))
        
        #cmd = self.action_number_to_cmd[action]
        reward = 0
        site_elements = self.driver.get_site_elements()
        current_element = None

        if cmd == "WAIT":
            pass

        elif cmd in {"CHOOSE_FIRST_CLICK", "CHOOSE_FIRST_SELECT", "CHOOSE_FIRST_ENTER"}:
            self.chosen_type = self.cmd_to_element_type[cmd]
            if len(site_elements[self.chosen_type]) > 0:
                #current_element = site_elements[self.chosen_type][0]
                self.chosen_number = 0
            else:
                reward = negative_reward()

        elif cmd == "NEXT":
            if self.chosen_number is None:
                reward = negative_reward()
            elif self.chosen_type:
                if len(site_elements[self.chosen_type]) > self.chosen_number + 1:
                    self.chosen_number += 1
                    #current_element = site_elements[self.chosen_type][self.chosen_number]
                else:
                    reward = negative_reward()
            else:
                reward = negative_reward()

        elif cmd in {"CLICK", "ENTER", "SELECT"}:
            if self.chosen_number is None:
                reward = negative_reward()
            elif self.chosen_type != self.cmd_to_element_type[cmd]:
                reward = negative_reward()
            elif (self.chosen_type and self.chosen_number < len(site_elements[self.chosen_type])):
                #reward = positive_reward()
                current_element = site_elements[self.chosen_type][self.chosen_number]
                if current_element is None: # perhaps, the element has been already used
                    reward = negative_reward()  # prevent clicking the same element twice
                else:
                    reward = positive_reward()
                    
                    #print("{} {}".format(cmd, current_element))
                    if cmd == "CLICK":
                        self.driver.action_on_element("CLICK", current_element)
                        #self.driver.click(current_element)
                    elif cmd == "ENTER":
                        self.driver.action_on_element("ENTER", current_element, data="Hello world")
                        #self.driver.enter(current_element, data="Hello world")
                    elif cmd == "SELECT":
                        pass
            else:
                reward = negative_reward()

        done = self.have_winner() or len(self.legal_actions()) == 0

        #reward = 1 if self.have_winner() else 0
        if self.have_winner():
            reward = WIN_REWARD * (1 - 0.9*(self.step_count / MAX_STEPS))
            self.wins += 1
            
            self.history_steps.append(self.step_count)
            if len(self.history_steps) > 100:
                self.history_steps.popleft()
            avg_steps = np.mean(self.history_steps)
            
            print("{}-th win in {} steps [{:.1f}]; reward={:.4f}".format(
                self.wins, self.step_count, avg_steps, reward))
            self.last_num_steps = self.step_count
            self.step_count = 0

        return self.get_observation_float(), reward, done, {}

    def get_observation(self):
        # It should return the current state as a numpy array of the float32 type
        # i.e. whole necessary information from the current webpage
        # including the list of active elements and so on.
        # probably, some additional information.
        # It will be fed to neural network.

        #board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        #board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        #board_to_play = numpy.full((3, 3), self.player).astype(float)
        #return numpy.array([board_player1, board_player2, board_to_play])

        #site_elements = {'clickables': ['Sign', 'Currency', 'Skip'], 'selectables': [], 'enterables': ['Your email']}
        site_elements = self.driver.get_site_elements()
        de_type = {0: 'clickables', 1: 'selectables', 2: 'enterables'}
        lengths = [len(site_elements[k]) for k in site_elements]
        
        (hight, width) = self.env_size

        env_state = [[1 if i<lengths[j] and site_elements[de_type[j]][i] else 0 for i in range(width)] for j in range(len(lengths))]
        env_state = np.array(env_state, dtype=np.int32)
        int_state = np.zeros((hight, width), dtype=np.int32)
        de_type_to_number = {y:x for x,y in de_type.items()}

        if self.chosen_type is not None and self.chosen_number is not None:
            de_type_number = de_type_to_number.get(self.chosen_type)
            int_state[de_type_number, self.chosen_number] = 1

        if self.prv_state is None:
            self.prv_state = env_state

        state = np.vstack([env_state, self.prv_state, int_state])
        #state = np.expand_dims(state, axis=0) # 3-dim. array is required
        state = state.flatten()
        self.prv_state = env_state
        return state

    def get_observation_float(self):
        return np.array(self.get_observation(), dtype=np.float32)

    def legal_actions(self):
        legal = list(range(len(self.possible_actions)))
        return legal

    def have_winner(self):
        #observation = self.get_observation()
        #sum_obs = np.sum(observation[:self.env_size[0]*self.env_size[1]])
        #return True if sum_obs < 0.01 else False # if all_active_elements_have_been_clicked
        return self.driver.is_target_achieved()

    def render(self):
        print("Display the game observation")
        print(self.get_observation())

    def obs_size(self):
        return 3 * self.env_size[0] * self.env_size[1]

    def number_of_actions(self):
        return NUMBER_ACTIONS


if __name__ == "__main__":

    im = Webdriver_imitation()

    env = TestDriverEnviroment()
    print(env.get_observation())
    print(env.obs_size())
    print(env.legal_actions())
    env.step("CHOOSE_FIRST_CLICK")
    env.step("NEXT")
    env.step("CLICK")
    env.step("CLICK")
    print(env.get_observation())

    #print("\n\nTest game")
    #game = Game()
    #game.step("WAIT")
    #game.step("CHOOSE_FIRST_CLICK")
    #game.step("NEXT")
    #game.step("CLICK")
