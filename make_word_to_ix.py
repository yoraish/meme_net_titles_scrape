"""This script gets all the posts from the top posts on reddit, looks at how many times each exists, and then maps each word (of the top 542 to an ix

"""
import praw
import json

def gen_word_to_ix():
    # get all the top posts (generator)
    my_client_secret = 'OaANBYgl4H6ye_tYV1PW_IvoagA'
    my_user_agent = 'yoraish'
    my_client_id = 'j1AQxvMldaBhWQ'
    reddit = praw.Reddit(client_id=my_client_id, client_secret=my_client_secret, user_agent=my_user_agent)

    hot_posts = reddit.subreddit('science').hot(limit=10000)

    word_count_dict = {} # maps words to their frequency of appearance'

    counter = 0
    for post in hot_posts:
        title = (''.join([i for i in post.title if i.isalpha() or i == ' '])).lower()
        title_words = [word for word in title.split()]
        print(counter)
        counter +=1

        for word in title_words:
            if word not in {'a', 'the', 'i', 'is', 'of', 'and', 'for', 'you', 'it', 'me', 'with', 'that', 'at', 'from'}:

                if word in word_count_dict:
                    word_count_dict[word] +=1
                else:
                    word_count_dict[word] =1

    
    # got through the words in order, and for the first 542, put in the final dict
    word_to_ix = {} # maps words to an index in the word vector

    counter = 0
    num_words = 1000
    for word, count in sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True):
        if counter >= num_words:
            break
        print(word, count)

        word_to_ix[word] = counter

        counter +=1

    
    return word_to_ix


if __name__ == "__main__":
    word_to_ix = gen_word_to_ix()

    # save to json
    with open('word_to_ix.json', 'w') as db:
        json.dump(word_to_ix, db)