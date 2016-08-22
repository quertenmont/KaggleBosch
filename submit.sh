spark-submit --class kaggle.Bosch --master local[4] target/BoschClasses-1.0-jar-with-dependencies.jar localhost:2181 sparkConsummer TutorialTopic 1
#spark-submit --class lq.KafkaWordCountProducer --master local[4] target/LQClasses-1.0-jar-with-dependencies.jar localhost:9092 TutorialTopic 250 10
