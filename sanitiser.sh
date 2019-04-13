cd _episodes_rmd
jupyter nbconvert --to markdown *ipynb

# remove comments from markdown
# beginning of comment <!--
for f in *.md
do sed -i.RidiculousMacBackup s/\<\!--//g $f
done
rm *.RidiculousMacBackup

# step 2
# end of comment -->
for f in *.md
do sed -i.RidiculousMacBackup s/--\>//g $f
done
rm *.RidiculousMacBackup

# Remove newlines at top of file which break markdown ability to interpter yaml
# newlines at top of file
for f in *.md
do sed -i.RidiculousMacBackup '/./,$!d' $f
done
rm *.RidiculousMacBackup

# put the images in the correct place
for f in *md
do
	# change link reference in the markdown to point to where jekyll expects the image to be
	sed -i.RidiculousMacBackup s/${f%.*}_files\/\.\.\\/fig/g $f
	rm *.RidiculousMacBackup
	# move the figures there
    cd "${f%.*}_files" && echo "Entering into ${f%.*}_files" && rsync -avzP * ../../fig/ && cd ../ || { echo "Error: could not enter into ${f%.*}_files; check there are no images"; continue; }
    #for y in $(echo *.png)
    #do
    #    rsync -avzP ${y} ../fig/
    #done
done


# rsync the md to the episodes folder
rsync -avzP *md ../_episodes/
