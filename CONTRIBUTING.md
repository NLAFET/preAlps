PreAlps is developed and maintained on two repositories: "inria gitlab" (inria) and "nlafet github" (nlafet). All developments are made on "inria gitlab", while stable versions are updated on "nlafet github". If you are going to contribute to the project, make sure to create a contributing local environment by following this guide:

1. Get a copy of preAlps from inria, and add a nlafet remote repository.

# Get a copy of the latest version of preAlps from inria
```
  $git clone https://gitlab.inria.fr/alps/preAlps.git preAlps  
```

# Add nlafet remote repository
```
 git remote add nlafet https://github.com/NLAFET/preAlps.git  
```

# Check the list of remote repositories  (inria and nlafet)
```
 git remote  
```
$git remote
nlafet
origin

# Check the url of the remote repositories (inria and nlafet)
```
 git remote -v
```
$git remote -v
nlafet	https://github.com/NLAFET/preAlps.git (fetch)
nlafet	https://github.com/NLAFET/preAlps.git (push)
origin	https://gitlab.inria.fr/alps/preAlps.git (fetch)
origin	https://gitlab.inria.fr/alps/preAlps.git (push)


2. Working with origin repository.

# Pull the latest version from origin (inria)
```
 git pull origin master
```
OR
```
 git pull
```

# Push the latest changes to the origin (inria)
```
 git push origin master
```

OR

```
 git push
```

3. In order to push later changes on nlafet, first push the latest version on "inria gitlab" (step 2), then get the latest version of "nlafet github" and push.

# Pull the latest version from nlafet
```
 git pull nlafet master
```

# Push the latest version on nlafet
```
 git push nlafet master
```
