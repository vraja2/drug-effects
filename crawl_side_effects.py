from xgoogle.search import GoogleSearch, SearchError

try:
    gs = GoogleSearch("aspirin side effects")
    gs.results_per_page = 50
    results = gs.get_results()
    urls = [res.url.encode("utf8") for res in results]
except SearchError, e:
    print "Search failed: %s" % e
