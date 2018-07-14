# the example of tensorflow high level `tf.estimator.Estimator` API distributed implement.
**fix with two error.**

master node error.
ValueError: If "cluster" is set in TF_CONFIG, it must have one "chief" node.
InvalidArgumentError: /job:worker/replica:0/task:0/device:CPU:0 unknown device.

``` python
        # line 280-309
        # update to master node
        # cluster must including a master node,
        # and master node must be the first worker node,
        # and each next worker node index subtract 1.
        # if not contain a master node, it will raise value error:
        # ValueError: If "cluster" is set in TF_CONFIG, it must have one "chief" node.
        # if the first worker node is not master node, it will raise value error:
        # InvalidArgumentError: /job:worker/replica:0/task:0/device:CPU:0 unknown device.
        # if you specific the master node, just comment the condition code.
        if FLAGS.job_name == "worker":
            if FLAGS.task_index == 0:
                tf.logging.info("update! worker to master, task index = 0.")
                FLAGS.job_name = "master"
            else:
                tf.logging.info("update! worker index subtract 1.")
                FLAGS.task_index = FLAGS.task_index - 1

        # distribution configuration
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")

        # cluster
        cluster = dict()
        cluster["ps"] = ps_hosts

        if len(worker_hosts) == 1:
            cluster["master"] = worker_hosts[:1]
        else:
            cluster["master"] = worker_hosts[:1]
            cluster["worker"] = worker_hosts[1:]
```

can't evaluate error.
ValueError: Could not find trained model in model_dir: {your_model_dir}.

``` python
        # line 254-275
        # master do evaluation
        # you should set eval_steps is 1 or smaller,
        # and set eval_batch_size is more bigger.
        # if it will evaluate much steps,
        # for cheackpoint life circle, it only keep 5 last .ckpt for default(you can customize).
        # and the next step's batch can't to evaluate.
        # and it will raise a error:
        # ValueError: Could not find trained model in model_dir: {your_model_dir}.
        if FLAGS.job_name == "master":
            try:
                results = model.evaluate(input_fn=lambda: input_fn(
                    FLAGS.test_data, 1, False, _NUM_EXAMPLES['validation']), steps=1)

                # Display evaluation metrics
                tf.logging.info("Results at epoch {0}".format((n + 1) * FLAGS.epochs_per_eval))
                tf.logging.info("================================================================")

                for key in sorted(results):
                    tf.logging.info("{0}: {1:.4f}".format(key, results[key]).replace(".0000", ""))
            except Exception as e:
                tf.logging.info("""================================================================\n
                {0}\n================================================================""".format(str(e)))
```

and update the running command.

``` bash
python wide_deep.py \
    --model_dir=${model_dir} \
    --ps_hosts="$PS" \
    --worker_hosts="$WORKER" \
    --job_name="$JOB_NAME" \
    --task_index="$TASK_INDEX"
```

final, thanks @Lapis-Hong 's work!

# Predicting Income with the Census Income Dataset
## Overview
The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) contains over 48,000 samples with attributes including age, occupation, education, and income (a binary label, either `>50K` or `<=50K`). The dataset is split into roughly 32,000 training and 16,000 testing samples.

Here, we use the [wide and deep model](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) to predict the income labels. The **wide model** is able to memorize interactions with data with a large number of features but not able to generalize these learned interactions on new data. The **deep model** generalizes well but is unable to learn exceptions within the data. The **wide and deep model** combines the two models and is able to generalize while learning exceptions.

For the purposes of this example code, the Census Income Data Set was chosen to allow the model to train in a reasonable amount of time. You'll notice that the deep model performs almost as well as the wide and deep model on this dataset. The wide and deep model truly shines on larger data sets with high-cardinality features, where each feature has millions/billions of unique possible values (which is the specialty of the wide model).

---

The code sample in this directory uses the high level `tf.estimator.Estimator` API. This API is great for fast iteration and quickly adapting models to your own datasets without major code overhauls. It allows you to move from single-worker training to distributed training, and it makes it easy to export model binaries for prediction.

The input function for the `Estimator` uses `tf.contrib.data.TextLineDataset`, which creates a `Dataset` object. The `Dataset` API makes it easy to apply transformations (map, batch, shuffle, etc.) to the data. [Read more here](https://www.tensorflow.org/programmers_guide/datasets).

The `Estimator` and `Dataset` APIs are both highly encouraged for fast development and efficient training.

## Running the code
### Setup
The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample uses for training is hosted by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/). We have provided a script that downloads and cleans the necessary files.

```
python data_download.py
```

This will download the files to `/tmp/census_data`. To change the directory, set the `--data_dir` flag.

### Training
You can run the code locally as follows:

```
python wide_deep.py
```

The model is saved to `/tmp/census_model` by default, which can be changed using the `--model_dir` flag.

To run the *wide* or *deep*-only models, set the `--model_type` flag to `wide` or `deep`. Other flags are configurable as well; see `wide_deep.py` for details.

The final accuracy should be over 83% with any of the three model types.

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=/tmp/census_model
```

## Additional Links

If you are interested in distributed training, take a look at [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

You can also [run this model on Cloud ML Engine](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction), which provides [hyperparameter tuning](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#hyperparameter_tuning) to maximize your model's results and enables [deploying your model for prediction](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#deploy_a_model_to_support_prediction).
