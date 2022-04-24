import environment, actor

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
# import logging
import sys
import pfrl
import torch
# logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

tokenizer = AutoTokenizer.from_pretrained("huggingtweets/elonmusk")  
model = AutoModelWithLMHead.from_pretrained("huggingtweets/elonmusk")
model.eval()
model.cuda()

sentiment = pipeline('sentiment-analysis',model="cardiffnlp/twitter-roberta-base-sentiment",tokenizer="cardiffnlp/twitter-roberta-base-sentiment",device=0,return_all_scores=True)

# transformers_logger = logging.getLogger('transformers')
# transformers_logger.setLevel(logging.CRITICAL)

sentiment("dogecoin is bad")

sentiment("dogecoin is bad")[0][0]['score']

class MyRLEnv(environment.TweetRLEnv):
    def get_reward(self, input_text, predicted_list, finish): # predicted will be the list of predicted token
      reward = 0
      if finish:
        if 1 < len(predicted_list) < 50:
          predicted_text = tokenizer.convert_tokens_to_string(predicted_list)
          # inverse perplexity
          inputs = tokenizer(input_text+predicted_text,return_tensors='pt').to('cuda')
          reward += (1/(torch.exp(model(**inputs, labels=inputs["input_ids"]).loss).mean().item()))
          # sentiment classifier
          reward += sentiment(predicted_text)[0][0]['score']
      return reward


observaton_list = ['i think dogecoin is']

env = MyRLEnv(model, tokenizer, observation_input=observaton_list)
actor = actor.TweetRLActor(env,model,tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=2000, epochs=20)

actor.predict('i think dogecoin is')

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=500,
    eval_n_steps=None,
    eval_n_episodes=1,       
    train_max_episode_len=50,  
    eval_interval=10,
    outdir='elon_musk_dogecoin', 
)

agent.load("./elon_musk_dogecoin/best")

actor.predict('i think dogecoin is')

