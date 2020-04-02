# GDrive link is here
Palm's project directory: https://drive.google.com/open?id=1OAZB82wyWc5GmFSatwja3AdTBPsFa69x

Golf's March 9th images: https://drive.google.com/file/d/1aCXR5tJwtUuyBrEh713EgvcxFq4xGUf7/view

Golf's March 13th images: https://drive.google.com/file/d/15XjKKXViz58s25moz76e2HBPRWTn93PE/view

## Note
The `predefined_classes.txt` for use in labelImg is in anns/

# Plan
## Query image selection method
|No|Train|Query|Result|
|---|---|---|---|
|1|All|All|---|
|2|All|Random Selected|---|
|3|Random Selected|Random Selected|---|
|4|Average Nearest Selected|Average Nearest Selected|---|

## Contrastive loss and distant threshold
|No|Train loss|distant threshold|result|
|---|---|---|---|
|1|Euclidean Distant|1.0|--|
|2|Invert Cosine Similarity|cosine(0.5)|--|
|3|Euclidean Distant|0.7|--|
