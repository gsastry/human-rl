from humanrl.classifier_tf import *
import multiprocessing
import numpy as np
import time
import humanrl.pong_catastrophe
import pickle
import subprocess
import argparse

input_path = "logs/pong_test"
save_classifier_path = "logs/pong_test/classifier_results"


def run(cmds, shell=True):
    return subprocess.run(cmds, shell=shell)

def generate_frames_a3c(test):
    sleep_time = 100 if test else 60*10    
    num_workers = 2 if test else 16
   
    run("""python train.py --num-workers {num_workers} --env-id Pong --log-dir {input_path} --catastrophe_reward 0""".
        format(num_workers=num_workers, input_path=input_path))
    run('sleep {}'.format(sleep_time))
    run('tmux kill-session -t a3c')



def run_classifier_metrics():
    with open(save_classifier_path + "/0/score_distribution.pkl", 'rb') as f:
        sd = pickle.load(f)

    print('\n\n Actual threshold', sd.threshold)
    print('Recall 0.9999. Threshold: ', threshold_from_predictions(sd.y_valid, sd.y_pred_valid,recall=0.9999))
    print('Recall 0.99. Threshold: ', threshold_from_predictions(sd.y_valid, sd.y_pred_valid, recall = 0.99))

 
    threshold = threshold_from_predictions(sd.y_train, sd.y_pred_train, recall=0.95)
    print('\n\nThreshold from training:', threshold)
    print('Metrics for training: ', classification_metrics(sd.y_train, sd.y_pred_train, threshold))

    threshold = threshold_from_predictions(sd.y_valid, sd.y_pred_valid)
    print('\n\nThreshold from validation:', threshold)
    print('Metrics for TEST:', classification_metrics(sd.y_test, sd.y_pred_test, threshold ))
 
    out_string = '''
    \n\nThreshold from training:' {threshold_train}
    Metrics for training: ' {metrics_train} 
    '\n\nThreshold from validation:' {threshold_test}
    'Metrics for TEST:' {metrics_test}
    '''.format(
        threshold_train=threshold_from_predictions(sd.y_train, sd.y_pred_train, recall=0.95),
        threshold_test = threshold_from_predictions(sd.y_valid, sd.y_pred_valid),
        metrics_train=classification_metrics(sd.y_train, sd.y_pred_train, threshold),
        metrics_test=classification_metrics(sd.y_test, sd.y_pred_test, threshold ))

    with open(save_classifier_path + "/0/score_distribution.txt", 'w') as f:
        f.write(out_string)

    
def train_classifier(test, blocker=False):
    
    number_train=20
    number_valid=30
    number_test=25

    steps = 1000
    batch_size= 1024
    conv_layers = 3

    if test:
        number_train=2
        number_valid=2
        number_test=2
        steps = 50
        batch_size = 20
        conv_layers = 2

    multiprocessing.freeze_support()

    episode_paths = frame.episode_paths(input_path)
    print('Found {} episodes'.format(len(episode_paths)))
    np.random.seed(seed=42)
    np.random.shuffle(episode_paths)

    if blocker:
        common_hparams = dict(use_action=True,  expected_positive_weight=0.05)
        labeller = humanrl.pong_catastrophe.PongBlockerLabeller()
    else:
        common_hparams = dict(use_action=False)
        labeller = humanrl.pong_catastrophe.PongClassifierLabeller()
        
    data_loader = DataLoader(labeller, TensorflowClassifierHparams(**common_hparams))
    datasets = data_loader.split_episodes(episode_paths,
                                          number_train, number_valid, number_test, use_all=False)


    hparams_list = [
        dict(image_crop_region=((34,34+160),(0,160)), #image_shape=[42, 42, 1], 
             convolution2d_stack_args=[(4, [3, 3], [2, 2])] * conv_layers, batch_size=batch_size, multiprocess=False,
             fully_connected_stack_args=[50,10],
             use_observation=False, use_image=True,
             verbose=True
         ) 
    ]

    start_experiment = time.time()
    print('Run experiment params: ', dict(number_train=number_train, number_valid=number_valid,
                                          number_test=number_test, steps=steps, batch_size=batch_size,
                                          conv_layers=conv_layers) )
    print('hparams', common_hparams, hparams_list[0])
    
    
    logdir = save_classifier_path
    run_experiments(
        logdir, data_loader, datasets, common_hparams, hparams_list, steps=steps, log_every=int(.1*steps))

    time_experiment = time.time() - start_experiment
    print('Steps: {}. Time in mins: {}'.format(steps, (1/60)*time_experiment))

    run_classifier_metrics()
    


def video_a3c_data():
    run("""python human_feedback.py -f {input_path} -n""".format(input_path=input_path))


def test_penalty_env(env):
    import envs
    env = envs.create_env("Pong", location="bottom", catastrophe_type="1", 
                          classifier_file=save_classifier_path + '/0/final.ckpt')
    
    import matplotlib.pyplot as plt

    observation = env.reset()
    
    for _ in range(20):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        plt.imshow(observation[:,:,0])
        plt.show()
        print('Cat: ', info['frame/is_catastrophe'])
        print('reward: ', reward)
        if done:
            break

def generate_frames_with_catastrophe_penalty():
    print('About to run a3c with cat penalty')
    run('sleep 200')
    
    #sleep_time = 200
    classifier_file = save_classifier_path + '/0/final.ckpt'
    logdir = 'logs/pong_test_with_penalty'
    run("""python train.py --num-workers 16 --env-id Pong --log-dir {logdir} --catastrophe_reward -1  --classifier_file {classifier_file}  --catastrophe_type 1""".format(logdir=logdir, classifier_file=classifier_file))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--blocker', action='store_true')
    parser.add_argument('--classifier', action='store_true')  ## Only run classifier
    args = parser.parse_args()

    ## TODO fix blocker mode

    if args.classifier:
        train_classifier(args.test, args.blocker)
    else:        
        generate_frames_a3c(args.test)
        train_classifier(args.test, args.blocker)
        generate_frames_with_catastrophe_penalty()


