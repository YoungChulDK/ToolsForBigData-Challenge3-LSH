# Challenge3 - Feature Hashing of 9700 Videos

**DTU - Computational Tools for Big Data - Week 12 + 13**

For this challenge you have to build your own locality sensitive hash function to hash small videos. You will get around 5GB (compressed) of videos (around 10k small videos). You have to write an LSH that takes a video and returns hash. You then have to cluster the videos using these hashes. How the dataset was constructed

The original data was 970 videos. Each video was processed to construct 10 new processed videos that are similar to the original, but has different modifications. The modifications include:
- Colors in the video might be have been made lighter or darker
- The speed of the video might be slower or faster
- There might be a border around the edge of some size
- The video might be a sub-clip of the original video (parts of beginning or end deleted)

The dataset is these 9700 processed videos.
 
**Downloading the data**
- The videos are available here: https://www.dropbox.com/s/8rs04nhptd8s61q/videos.zip

**Showing how good the solution is**

This script (https://gist.github.com/utdiscant/213b6c1bdad5f93bd5d1ca1a7eece375) calculates the so-called adjusted Rand Index of two clusterings (https://en.wikipedia.org/wiki/Rand_index). The input for the function in the script is a list of sets, where each set corresponds to a cluster and contains names of video files without the ending (eg. DMDR1U2RA7VN). The truth-variable in the script contains the actual clustering, and your clustering will be compared to this clustering.

To show how good your solution is, compute the rand-index between your clustering and the truth using that function. You should of course not use the truth when building your hash-functions, but you can use the rand-index calculation to see how you are doing.

**Showing how fast the solution is**
To measure the speed of your solution, time how long time it takes to hash all videos and create a clustering. This includes loading videos from disk and constructing the clustering. It does not include testing how good the solution is.

**Writing the report**
- Solve the problem and explain your approach
- Show how good your solution is
- Show how fast your solution is
