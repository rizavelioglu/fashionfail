### FAQ
**[Q]** _Why didn't we scrape the category information and avoid using a LLM?_
>The scraped categories do not align with the Fashionpedia ontology. As a result, manual category mapping is
necessary, which is time-consuming and prone to errors.

**[Q]** _Why only a subset of categories are used?_
>To make it possible to automatize the annotation pipeline, which is designed to provide only a single
annotation for an image. Providing multiple annotations per image automatically is extremely challenging. For example,
consider a shoe image with an applique on it which has 2 annotations; _"applique"_ and _"shoe"_. Multiple issues may occur:
>- how to annotate label: "applique" can not be extracted from product descriptions,
>- GroundingDINO's performance differs for different labels: boxes for "applique" may exhibit less accuracy than those for "shoe",
>- mask ambiguity: SAM provides 3 masks for each box (to solve 'ambiguity') and we select the one with the highest score.
However, for small boxes (e.g. applique) the correct mask may not be the one with the highest score, hence, manual
labeling is required.

**[Q]** _Can SAM be used for fashion object segmentation, which would eliminate training a specific model for this task?_
>SAM is great at segmenting anything however, it does not classify the segmented masks. In addition, SAM
produces many masks (segmenting literally anything), for example, for a shoe image it may provide different masks for the shoelace,
applique, shoe sole, etc. which requires a further method to combine those masks into a single one.

**[Q]** _How can you publish scraped data?_
>According to German laws [(§ 51 UrhG.)](https://www.gesetze-im-internet.de/urhg/__51.html) image files can be
> published for scientific works.
> However, to ensure compliance and avoid any potential data privacy issues, we preferred to be overly careful
> by sharing only the URLs of the images.

**[Q]** _What are the future research directions?_
<span style="color:red">TODO</span>


_**Final remarks:**_
<span style="color:red">TODO</span>

---
<div style="display: flex; justify-content: space-between;">

   [Back](06_visualization.md)

   [Back to docs](README.md)

</div>
