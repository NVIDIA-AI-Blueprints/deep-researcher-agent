# DuckDuckGo News Search

NAT source package that exposes a DuckDuckGo news search tool backed by `ddgs`.

## Tool

```yaml
duckduckgo_news_search_tool:
  _type: duckduckgo_news_search
  max_results: 5
  timelimit: w
```

The tool returns lightweight document blocks with title, source, date, snippet,
and URL so Deep Researcher Agent citation capture can treat news results as citable sources.
