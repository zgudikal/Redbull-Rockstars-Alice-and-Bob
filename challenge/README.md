# Challenge

This challenge is broken into two notebooks:
1. `challenge/1-challenge.ipynb` 
2. `challenge/2-classical-to-quantum-codes.ipynb`

The first notebook contains all the tasks for this hackathon, while the second notebook serves as an essential resource for the core task of this challenge. **It is highly recommended that you read both notebooks.** We have also included additional notebooks in the `challenge/resources/` folder. These are not required readings, but provide deeper insight into particular topics.

Tasks within sections 1-4 are intended to bring participants up to speed on concepts within quantum error correction with biased physical qubits. The task in section 5 is the core open-ended challenge of this hackathon - we will call this task the "Core Task". 

As such, judges will award 30% of the total points based on answers to Tasks within sections 1-4 and 70% of the total points based on the submission to the Core Task in section 5. To help teams prioritize, we have indicated the exact number of points each subtask is worth out of a total of 100pts. 


# Judging Criteria

Here are the general guidelines by which judging will occur.

## Tasks in Section 1-4

These tasks are defined clearly, so judging will be based on:

1. Accuracy - Are the answers and calculations correct?
2. Completeness - Are all required questions, figures, and explanations provided?
3. Clarity - Are the results easy to follow and well explained?

## Core Task in Section 5

This task is intentionally open-ended. 

*As such, we will base our judging on:*
1. Technical Soundness - Are the methods correct, internally consistent, and based on reasonable assumptions?
2. Depth of Analysis - Does the submission go beyond surface-level results and explore behavior, limitations, and tradeoffs?
3. Originality - Does the work show independent thinking or creative adaptation, rather than a straightforward replication?
4. Impact and Relevance - Does the solution meaningfully address the core problem and demonstrate why it matters?
5. Clarity and Organization - Is the presentation of the work (i.e. notebooks, slides, etc.) easy to follow, well structured, and clearly communicated?

If done well, your work towards this core task can result in novel, publishable research of great relevance to Alice & Bob. So, be proud of your work!


# Submission

Your submission will consist of two parts:
1. Please make a pull request to the challenge repository. You should put all of your submission materials into one folder named as `team-<name>/` that lives in the root level of this repository, for example `team-cats/`. Please do not modify or delete the `challenge/` folder in this repository. This will allow us to actually merge your pull request to live on in the main branch! Your team folder should then contain all of your submission materials, including:
   1. A copy of `1-challenge.ipynb` filled with responses to Tasks in sections 1-4.
   2. We recommend putting code, papers, slides, README, etc. related to the Core Task in Section 5 in a separate sub-folder. It would help to write a README to describe the contents of your submission. We will review this code when deciding winners.
   3. So, the structure of your submission should look like:
      ```
      challenge/ # no modifications
      team-cats/
         1-challenge.ipynb
         core/
            README.md
            notebook.ipynb # not required, just an example
            slides.pdf # not required, just an example, a link to google slides in the README also suffices
            paper.pdf # not required, just an example
            ...
      ```
2. Please prepare for a 7 minute presentation and 3 minute Q&A with judges. You can keep editing your presentation until you give it. In this presentation, we recommend that you focus on your results related to the Core Task.

# Support

The mentors for this challenge are:
* Shantanu Jha - a PhD student building error corrected bosonic qubits at MIT
* Diego Polimeni - a researcher of fault-tolerant compilation and hardware integration at RheinMain University 

Please do not hesitate to come to us with any questions you may have. We are here to help you.