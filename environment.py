import random, gym, torch, numpy


class TweetRLEnv(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[]):
        vocabs = list(dict(sorted(tokenizer.vocab.items(), key=lambda item: item[1])).keys())
        self.action_space = gym.spaces.Discrete(len(vocabs))
        self.actions = vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.target_table = {}
        self.input_text = ""
        self.predicted = []
        self.reset()

    def step(self, action):
        if isinstance(action, numpy.ndarray):
            action = numpy.argmax(action)
        predicted, finish, predicted_str = self.predict(vocab_id=action)
        reward = self.get_reward(self.input_text, predicted, finish)
        self.predicted = predicted
        return self.get_obs(predicted), reward, finish, {"predicted_str": predicted_str}

    def get_reward(self, input_text, predicted_list, finish):
        reward = 1
        return reward

    def reset(self, input_text=None):
        self.predicted = []
        if input_text is None:
            self.input_text = random.choice(self.observation_space)
        else:
            self.input_text = input_text
        return self.get_obs()

    def get_obs(self, predicted=[]):
        with torch.no_grad():
            p_text = self.tokenizer.convert_tokens_to_string(predicted)
            if len([k for k, v in self.model.named_parameters() if 'decoder' in k]) > 0:
                feature_dict = self.tokenizer([self.input_text], return_tensors='pt', add_special_tokens=False).to(
                    self.model.device)
                predicted = [self.tokenizer.eos_token] + predicted
                dec_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(predicted)]).to(self.model.device)
                feature_dict['decoder_input_ids'] = dec_input
                prediction = self.model(**feature_dict, output_hidden_states=True)
                outputs = prediction.decoder_hidden_states[-1].squeeze(0)
            else:
                feature_dict = self.tokenizer([[self.input_text, p_text]], return_tensors='pt',
                                              add_special_tokens=False).to(self.model.device)
                prediction = self.model(**feature_dict, output_hidden_states=True)
                outputs = prediction.hidden_states[-1].squeeze(0)
            return outputs.data[-1]

    def predict(self, vocab_id):
        predicted = self.predicted
        with torch.no_grad():
            pred_word = self.actions[vocab_id]
            model_max_length = max(self.model.config.max_length, self.tokenizer.model_max_length)
            if pred_word == self.tokenizer.sep_token or pred_word == self.tokenizer.eos_token \
                    or len(pred_word) < 1 or len(self.predicted) > model_max_length:
                return predicted, True, self.tokenizer.convert_tokens_to_string(predicted)
            else:
                predicted += [pred_word]
                return predicted, False, self.tokenizer.convert_tokens_to_string(predicted)