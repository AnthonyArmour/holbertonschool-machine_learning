[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# QA Bot Using Bert Transformer a Universal Word Encoder for Embeddings
I can't share the corpus that was provided for this project by the school.

## Dependencies
| Library/Framework         | Version |
| ------------------------- | ------- |
| Python                    | ^3.7.3  |
| numpy                     | ^1.19.5 |
| tensorflow                | ^2.6.0  |
| tensorflow-hub            | ^0.12.0 |
| transformers              | ^4.17.0 |

## Tasks

### [Question Answer](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-qa_bot/2-qa.py "Question Answer")
Answers a question, given a reference text.
``` python
#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
```

```
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
```

### [Semantic Search](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-qa_bot/3-semantic_search.py "Semantic Search")
Performs semantic search on a corpus of documents.
``` python
#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

# corpus_path = ZendeskArticles

print(semantic_search('corpus_path', 'When are PLDs?'))
```

```
PLD Overview
Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
PLD Basics
PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
When a peer has a question, rather than offering the solution, ask the following:
"How did you come to that conclusion?"
"What have you tried?"
"Did the man page give you a lead?"
"Did you think about this concept?"
Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
```

### [QA Bot](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x03-qa_bot/4-qa.py "QA Bot")
Answers questions from a corpus of multiple reference texts.
``` python
#!/usr/bin/env python3

question_answer = __import__('4-qa').question_answer

# corpus_path = ZendeskArticles

question_answer('corpus_path')
```

```
Q: When are PLDs?
A: on - site days from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
```