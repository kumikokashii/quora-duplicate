import pickle
import logging
logging.getLogger('tensorflow').setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from sklearn.utils import shuffle
import datetime
from scipy.special import expit

def prep_logloss_log(naming=None, variables=None):
    logloss_log = {}
    logloss_log['variables'] = variables
            
    logloss_log['pickle_file'] = 'pickle_' + naming
    logloss_log['tb_log_directory'] = 'tb_' + naming
    logloss_log['checkpoint_directory'] = 'ckpt_' + naming  

    #Create a pickle file
    pickle.dump(logloss_log, open(logloss_log['pickle_file'], 'wb'))

    #Create TensorBoard and checkpoint directories
    for d in (logloss_log['tb_log_directory'], logloss_log['checkpoint_directory']):
        if tf.gfile.Exists(d):
            tf.gfile.DeleteRecursively(d)
        tf.gfile.MakeDirs(d)

    tf.gfile.MkDir(logloss_log['checkpoint_directory'] + '/best')
    tf.gfile.MkDir(logloss_log['checkpoint_directory'] + '/hourly')
    
    return logloss_log

def batch_normalize(input_tensor, is_training, global_step, scope):
    train_first = tf.logical_and(is_training, tf.equal(global_step, 0))
    return tf.cond(train_first, 
                   lambda: batch_norm(input_tensor, is_training=is_training, center=False, 
                                      updates_collections=None, scope=scope, reuse=None), 
                   lambda: batch_norm(input_tensor, is_training=is_training, center=False, 
                                      updates_collections=None, scope=scope, reuse=True))

def weight_variable(shape, weights_stddev, name):
    initial = tf.truncated_normal(shape, stddev=weights_stddev)
    return tf.Variable(initial, name=name)

def bias_variable(biases_initial, shape, name):
    initial = tf.constant(biases_initial, shape=shape)
    return tf.Variable(initial, name=name)
    
def train_model(graph_variables, logloss_log, train_data, train_labels, validation_data, validation_labels):
    with tf.Session(graph=graph_variables['graph']) as sess:   
        offset = 0
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(logloss_log['tb_log_directory'], sess.graph)
        saver_best = tf.train.Saver()
        saver_hourly = tf.train.Saver(max_to_keep=None)  
        last_hourly_save = datetime.datetime.now()
        
        start_step = 0
        logloss_log['index'] = []
        logloss_log['steps'] = []
        logloss_log['training'] = []
        logloss_log['validation'] = []
        logloss_log['ave_validation'] = []
        logloss_log['end_time'] = []
        logloss_log['min_ave_validation_logloss'] = {'step': -1, 'logloss': 1000000, 'patient_till': -1}
            
        for i in range(start_step, logloss_log['variables']['max_steps']):
            if i == start_step:
                logloss_log['train_start'] = datetime.datetime.now()
                print('%-10s%s%s' % ('TRAINING ', ' START @ ', logloss_log['train_start']))
                
            offset = offset % logloss_log['variables']['train_data_size']
            #Shuffle every epoch
            if offset == 0:
                train_data, train_labels = shuffle(train_data, train_labels)

            if offset <= (logloss_log['variables']['train_data_size'] - logloss_log['variables']['batch_size']):
                batch_data = train_data[offset: offset+logloss_log['variables']['batch_size'], :]
                batch_labels = train_labels[offset: offset+logloss_log['variables']['batch_size']]
                offset += logloss_log['variables']['batch_size']
            else:
                batch_data = train_data[offset: logloss_log['variables']['train_data_size'], :]
                batch_labels = train_labels[offset: logloss_log['variables']['train_data_size']]
                offset = 0
            _, summary = sess.run([graph_variables['optimizer'], graph_variables['summarizer']], 
                                  feed_dict={graph_variables['data']: batch_data, graph_variables['labels']: batch_labels,
                                             graph_variables['keep_prob']: logloss_log['variables']['dropout_train_keep_prob'], graph_variables['is_training']: True})

            if i % logloss_log['variables']['log_every'] == 0:
                writer.add_summary(summary, i)
                training_logloss = graph_variables['logloss'].eval(feed_dict={graph_variables['data']: batch_data, graph_variables['labels']: batch_labels, 
                                                             graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False})
                validation_logloss = graph_variables['logloss'].eval(feed_dict={graph_variables['data']: validation_data, graph_variables['labels']: validation_labels, 
                                                             graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False})
                
                logloss_log['index'].append(len(logloss_log['index']))
                logloss_log['steps'].append(i)
                logloss_log['training'].append(training_logloss)
                logloss_log['validation'].append(validation_logloss)
                if len(logloss_log['validation']) < logloss_log['variables']['average_n_validation_logloss']:
                    start = 0
                    num = len(logloss_log['validation'])
                else:
                    start = len(logloss_log['validation']) - logloss_log['variables']['average_n_validation_logloss']
                    num = logloss_log['variables']['average_n_validation_logloss']
                n_recent_validation_logloss_log = logloss_log['validation'][start: start+num]
                ave_validation_logloss = sum(n_recent_validation_logloss_log)/float(num)
                logloss_log['ave_validation'].append(ave_validation_logloss)
                logloss_log['end_time'].append(datetime.datetime.now())

                #Save a model and a pickle file every hour
                if last_hourly_save + datetime.timedelta(hours=1) < datetime.datetime.now():                    
                    path_checkpoint_file = saver_hourly.save(sess, logloss_log['checkpoint_directory'] + '/hourly/model', global_step=i, latest_filename='hourly_checkpoint')

                    pickle.dump(logloss_log, open(logloss_log['pickle_file'], 'wb'))
                    print('<< hourly save at %s and %s >>' % (path_checkpoint_file, logloss_log['pickle_file']))
                    last_hourly_save = datetime.datetime.now()
                
                #If it is the best model so far
                if ave_validation_logloss <= logloss_log['min_ave_validation_logloss']['logloss']:
                    logloss_log['min_ave_validation_logloss']['step'] = i
                    logloss_log['min_ave_validation_logloss']['logloss'] = ave_validation_logloss
                    logloss_log['min_ave_validation_logloss']['patient_till'] = ave_validation_logloss + logloss_log['variables']['early_stopping_patience']

                    path_checkpoint_file = saver_best.save(sess, logloss_log['checkpoint_directory'] + '/best/model', global_step=i, latest_filename='best_checkpoint')
                    print('<< best model so far is saved in %s >> ave validation logloss %7.5f' % 
                          (path_checkpoint_file, ave_validation_logloss))
                    pickle.dump(logloss_log, open(logloss_log['pickle_file'], 'wb'))

                #If not, and if patience is over
                elif (i > logloss_log['variables']['start_step_early_stopping']) & (ave_validation_logloss > logloss_log['min_ave_validation_logloss']['patient_till']):
                    print('** reached early stopping patience **')
                    break

            if i % logloss_log['variables']['print_every'] == 0:
                print('STEP %7d%s%s'%(i, ' END @ ', datetime.datetime.now()) +
                      ', training logloss %7.5f, ave validation logloss %7.5f' % (training_logloss, ave_validation_logloss))

        logloss_log['train_end'] = datetime.datetime.now()
        print('%-12s%s%s' % ('TRAINING ', ' END @ ', logloss_log['train_end']))

def predict(graph_variables, logloss_log, test_data):
    with tf.Session(graph=graph_variables['graph']) as sess:
        tf.global_variables_initializer().run()
        #Grab the model with the highest ave validation logloss to evaluate test logloss
        logloss_log['best_model'] = tf.train.latest_checkpoint(logloss_log['checkpoint_directory'] + '/best', 
                                                               latest_filename='best_checkpoint')
        saver = tf.train.Saver()
        saver.restore(sess, logloss_log['best_model'])
        
        test_logloss = graph_variables['logits'].eval(feed_dict={graph_variables['data']: test_data,  
                                                                 graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False})
        
        return expit(test_logloss.flatten())