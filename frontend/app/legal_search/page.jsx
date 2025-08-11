"use client"

import { useState, useEffect } from "react"
import {
  Search,
  Scale,
  FileText,
  Calendar,
  Filter,
  BookOpen,
  Download,
  ExternalLink,
  Clock,
  Building,
  AlertCircle,
  CheckCircle,
  Loader,
  Star,
  Eye,
  Bookmark,
  Share2,
  X,
  ChevronDown,
  ChevronUp,
  Copy,
  RefreshCw,
  ArrowRight,
  Award,
  Zap,
  Gavel,
} from "lucide-react"

const API_BASE_URL = "http://localhost:5000"

const LegalResearchApp = () => {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchType, setSearchType] = useState("all")
  const [searchResults, setSearchResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState({
    doctypes: "",
    fromdate: "",
    todate: "",
    court_type: "",
    days: 30,
  })
  const [aiAnalysis, setAiAnalysis] = useState(null)
  const [analyzingWithAI, setAnalyzingWithAI] = useState(false)
  const [legalNews, setLegalNews] = useState([])
  const [courtTypes, setCourtTypes] = useState({})
  const [activeTab, setActiveTab] = useState("search")
  const [savedSearches, setSavedSearches] = useState([])
  const [bookmarkedCases, setBookmarkedCases] = useState([])
  const [selectedCase, setSelectedCase] = useState(null)
  const [showCaseModal, setShowCaseModal] = useState(false)
  const [expandedResults, setExpandedResults] = useState({})
  const [downloadingCase, setDownloadingCase] = useState(null)

  useEffect(() => {
    fetchCourtTypes()
    fetchLegalNews()
    loadSavedData()
  }, [])

  const loadSavedData = () => {
    const saved = localStorage.getItem("savedSearches")
    const bookmarked = localStorage.getItem("bookmarkedCases")
    if (saved) setSavedSearches(JSON.parse(saved))
    if (bookmarked) setBookmarkedCases(JSON.parse(bookmarked))
  }

  const saveSearch = (query, results) => {
    const newSearch = {
      id: Date.now(),
      query,
      timestamp: new Date().toISOString(),
      resultCount: results?.sources ? Object.keys(results.sources).length : 0,
    }
    const updatedSaves = [newSearch, ...savedSearches.slice(0, 9)]
    setSavedSearches(updatedSaves)
    localStorage.setItem("savedSearches", JSON.stringify(updatedSaves))
  }

  const bookmarkCase = (caseData) => {
    const bookmark = {
      id: Date.now(),
      ...caseData,
      bookmarkedAt: new Date().toISOString(),
    }
    const updatedBookmarks = [bookmark, ...bookmarkedCases.slice(0, 49)]
    setBookmarkedCases(updatedBookmarks)
    localStorage.setItem("bookmarkedCases", JSON.stringify(updatedBookmarks))

    // Show success notification
    const notification = document.createElement("div")
    notification.className =
      "fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 animate-pulse"
    notification.textContent = "Case bookmarked successfully!"
    document.body.appendChild(notification)
    setTimeout(() => document.body.removeChild(notification), 3000)
  }

  const fetchCourtTypes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/courts`)
      const data = await response.json()
      setCourtTypes(data)
    } catch (error) {
      console.error("Error fetching court types:", error)
    }
  }

  const fetchLegalNews = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/news?days=7`)
      const data = await response.json()
      setLegalNews(data.news || [])
    } catch (error) {
      console.error("Error fetching legal news:", error)
    }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!searchQuery.trim()) return

    setIsLoading(true)
    setSearchResults(null)
    setAiAnalysis(null)

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: searchQuery,
          search_type: searchType,
          filters: filters,
        }),
      })

      const data = await response.json()
      setSearchResults(data)
      saveSearch(searchQuery, data)
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults({ error: "Failed to fetch search results" })
    } finally {
      setIsLoading(false)
    }
  }

  const analyzeWithAI = async () => {
    if (!searchResults) return
    setAnalyzingWithAI(true)

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          search_results: searchResults,
          analysis_type: "comprehensive",
        }),
      })

      const data = await response.json()
      setAiAnalysis(data)
    } catch (error) {
      console.error("AI Analysis error:", error)
    } finally {
      setAnalyzingWithAI(false)
    }
  }

  const viewFullCase = async (caseData) => {
    setSelectedCase(caseData)
    setShowCaseModal(true)
  }

  const downloadCase = async (caseData) => {
    setDownloadingCase(caseData.tid)

    try {
      // Simulate API call to get full case content
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Create downloadable content
      const content = `
CASE TITLE: ${caseData.title}

HEADLINE: ${caseData.headline}

SOURCE: ${caseData.docsource}

DOCUMENT ID: ${caseData.tid}

CITATIONS: ${caseData.citeList ? caseData.citeList.join(", ") : "None"}

FULL TEXT:
${caseData.headline}

---
Downloaded from Indian Legal Research Platform
Generated on: ${new Date().toLocaleString()}
      `

      const blob = new Blob([content], { type: "text/plain" })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${caseData.title.replace(/[^a-z0-9]/gi, "_").toLowerCase()}.txt`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)

      // Show success notification
      const notification = document.createElement("div")
      notification.className = "fixed top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg z-50"
      notification.textContent = "Case downloaded successfully!"
      document.body.appendChild(notification)
      setTimeout(() => document.body.removeChild(notification), 3000)
    } catch (error) {
      console.error("Download error:", error)
    } finally {
      setDownloadingCase(null)
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
    const notification = document.createElement("div")
    notification.className = "fixed top-4 right-4 bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg z-50"
    notification.textContent = "Copied to clipboard!"
    document.body.appendChild(notification)
    setTimeout(() => document.body.removeChild(notification), 2000)
  }

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString("en-IN", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const toggleResultExpansion = (source) => {
    setExpandedResults((prev) => ({
      ...prev,
      [source]: !prev[source],
    }))
  }

  const SearchHeader = () => (
    <div className="relative bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 text-white overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-20">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fillRule='evenodd'%3E%3Cg fill='%239C92AC' fillOpacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
            backgroundSize: "60px 60px",
          }}
        ></div>
      </div>
      \
      <div className="relative max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <Scale className="h-16 w-16 text-amber-400 drop-shadow-lg" />
              <div className="absolute -top-1 -right-1 w-6 h-6 bg-green-400 rounded-full flex items-center justify-center">
                <CheckCircle className="h-4 w-4 text-white" />
              </div>
            </div>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
            Indian Legal Research
          </h1>
          <p className="text-xl md:text-2xl text-blue-200 max-w-3xl mx-auto leading-relaxed">
            Comprehensive legal research across Supreme Court, High Courts, Tribunals, and legal news with AI-powered
            insights
          </p>
        </div>

        <form onSubmit={handleSearch} className="max-w-5xl mx-auto">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/20">
            <div className="flex flex-col lg:flex-row gap-4 mb-6">
              <div className="flex-1 relative group">
                <Search className="absolute left-4 top-4 h-6 w-6 text-gray-400 group-focus-within:text-blue-500 transition-colors" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search cases, acts, judgments, legal precedents..."
                  className="w-full pl-12 pr-4 py-4 text-gray-900 bg-white rounded-xl border-2 border-gray-200 focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 text-lg placeholder-gray-500 transition-all duration-200"
                />
              </div>
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value)}
                className="px-6 py-4 text-gray-900 bg-white rounded-xl border-2 border-gray-200 focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 min-w-[220px] font-medium"
              >
                <option value="all">🔍 All Sources</option>
                <option value="cases">⚖️ Cases & Judgments</option>
                <option value="supreme_court">🏛️ Supreme Court</option>
                <option value="high_courts">🏢 High Courts</option>
                <option value="tribunals">📋 Tribunals</option>
                <option value="news">📰 Legal News</option>
              </select>
              <button
                type="submit"
                disabled={isLoading}
                className="px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-black font-bold rounded-xl disabled:opacity-50 flex items-center justify-center min-w-[140px] shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
              >
                {isLoading ? (
                  <Loader className="animate-spin h-6 w-6" />
                ) : (
                  <>
                    <Search className="h-5 w-5 mr-2" />
                    Search
                  </>
                )}
              </button>
            </div>

            <div className="flex flex-wrap justify-center gap-4">
              <button
                type="button"
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center px-6 py-3 bg-blue-600/80 hover:bg-blue-600 rounded-xl text-sm font-medium backdrop-blur-sm transition-all duration-200 hover:scale-105"
              >
                <Filter className="h-4 w-4 mr-2" />
                {showFilters ? "Hide" : "Show"} Filters
                {showFilters ? <ChevronUp className="h-4 w-4 ml-2" /> : <ChevronDown className="h-4 w-4 ml-2" />}
              </button>
              {searchResults && (
                <button
                  onClick={analyzeWithAI}
                  disabled={analyzingWithAI}
                  className="flex items-center px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 rounded-xl text-sm font-medium disabled:opacity-50 transition-all duration-200 hover:scale-105"
                >
                  {analyzingWithAI ? (
                    <Loader className="animate-spin h-4 w-4 mr-2" />
                  ) : (
                    <Zap className="h-4 w-4 mr-2" />
                  )}
                  AI Analysis
                </button>
              )}
            </div>

            {showFilters && (
              <div className="mt-8 bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-lg font-semibold mb-6 text-white flex items-center">
                  <Filter className="h-5 w-5 mr-2" />
                  Advanced Filters
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-3 text-gray-200">Court Type</label>
                    <select
                      value={filters.doctypes}
                      onChange={(e) => setFilters({ ...filters, doctypes: e.target.value })}
                      className="w-full px-4 py-3 text-gray-900 bg-white rounded-lg border-2 border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">All Courts</option>
                      {Object.entries(courtTypes.indian_kanoon_doctypes || {}).map(([key, value]) => (
                        <option key={key} value={key}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-3 text-gray-200">From Date</label>
                    <input
                      type="date"
                      value={filters.fromdate}
                      onChange={(e) => setFilters({ ...filters, fromdate: e.target.value })}
                      className="w-full px-4 py-3 text-gray-900 bg-white rounded-lg border-2 border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-3 text-gray-200">To Date</label>
                    <input
                      type="date"
                      value={filters.todate}
                      onChange={(e) => setFilters({ ...filters, todate: e.target.value })}
                      className="w-full px-4 py-3 text-gray-900 bg-white rounded-lg border-2 border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  )

  const TabNavigation = () => (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-40 shadow-sm">
      <div className="max-w-7xl mx-auto px-6">
        <nav className="flex space-x-1">
          {[
            {
              id: "search",
              label: "Search Results",
              icon: Search,
              count: searchResults?.sources ? Object.keys(searchResults.sources).length : 0,
            },
            { id: "analysis", label: "AI Analysis", icon: Zap, count: aiAnalysis ? 1 : 0 },
            { id: "news", label: "Legal News", icon: FileText, count: legalNews.length },
            {
              id: "saved",
              label: "Saved & Bookmarks",
              icon: Bookmark,
              count: savedSearches.length + bookmarkedCases.length,
            },
          ].map(({ id, label, icon: Icon, count }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`relative flex items-center py-4 px-6 font-medium text-sm transition-all duration-200 ${
                activeTab === id
                  ? "text-blue-600 border-b-2 border-blue-600 bg-blue-50"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
              }`}
            >
              <Icon className="h-5 w-5 mr-2" />
              {label}
              {count > 0 && (
                <span
                  className={`ml-2 px-2 py-1 text-xs font-bold rounded-full ${
                    activeTab === id ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-700"
                  }`}
                >
                  {count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>
    </div>
  )

  const CaseModal = () => {
    if (!showCaseModal || !selectedCase) return null

    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
          <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
            <div className="flex items-center">
              <Scale className="h-6 w-6 text-blue-600 mr-3" />
              <h2 className="text-xl font-bold text-gray-900">Case Details</h2>
            </div>
            <button
              onClick={() => setShowCaseModal(false)}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            <div className="space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-gray-900 mb-3">{selectedCase.title}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                    {selectedCase.docsource}
                  </span>
                  {selectedCase.tid && (
                    <span className="px-3 py-1 bg-gray-100 text-gray-800 rounded-full text-sm font-medium">
                      ID: {selectedCase.tid}
                    </span>
                  )}
                </div>
              </div>

              <div className="bg-gray-50 rounded-xl p-6">
                <h4 className="font-semibold text-gray-900 mb-3">Case Summary</h4>
                <p className="text-gray-700 leading-relaxed">{selectedCase.headline}</p>
              </div>

              {selectedCase.citeList && selectedCase.citeList.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-900 mb-3">Citations</h4>
                  <div className="bg-blue-50 rounded-xl p-4">
                    <div className="flex flex-wrap gap-2">
                      {selectedCase.citeList.map((citation, index) => (
                        <span key={index} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-lg text-sm">
                          {citation}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div className="flex flex-wrap gap-3 pt-4 border-t border-gray-200">
                <button
                  onClick={() => downloadCase(selectedCase)}
                  disabled={downloadingCase === selectedCase.tid}
                  className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
                >
                  {downloadingCase === selectedCase.tid ? (
                    <Loader className="animate-spin h-4 w-4 mr-2" />
                  ) : (
                    <Download className="h-4 w-4 mr-2" />
                  )}
                  Download
                </button>
                <button
                  onClick={() => copyToClipboard(selectedCase.title)}
                  className="flex items-center px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy Title
                </button>
                <button
                  onClick={() => bookmarkCase(selectedCase)}
                  className="flex items-center px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-colors"
                >
                  <Star className="h-4 w-4 mr-2" />
                  Bookmark
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const SearchResults = () => {
    if (!searchResults) {
      return (
        <div className="text-center py-16">
          <Search className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-600 mb-2">Ready to Search</h3>
          <p className="text-gray-500">Enter your search query above to find legal cases and documents</p>
        </div>
      )
    }

    if (searchResults.error) {
      return (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-6 mb-6">
          <div className="flex items-center">
            <AlertCircle className="h-6 w-6 text-red-600 mr-3" />
            <div>
              <h3 className="text-lg font-semibold text-red-800">Search Error</h3>
              <p className="text-red-600">{searchResults.error}</p>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className="space-y-8">
        <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <CheckCircle className="h-6 w-6 text-green-600 mr-3" />
              <div>
                <span className="font-semibold text-gray-900">Search completed for "</span>
                <span className="font-bold text-blue-600">{searchResults.query}</span>
                <span className="font-semibold text-gray-900">"</span>
              </div>
            </div>
            <div className="flex items-center text-sm text-gray-600 bg-white px-3 py-1 rounded-full">
              <Clock className="h-4 w-4 mr-1" />
              {formatDate(searchResults.timestamp)}
            </div>
          </div>
        </div>

        {Object.entries(searchResults.sources || {}).map(([source, data]) => (
          <ResultSection
            key={source}
            source={source}
            data={data}
            onBookmark={bookmarkCase}
            onViewFull={viewFullCase}
            onDownload={downloadCase}
            isExpanded={expandedResults[source]}
            onToggleExpansion={() => toggleResultExpansion(source)}
            downloadingCase={downloadingCase}
          />
        ))}
      </div>
    )
  }

  const ResultSection = ({
    source,
    data,
    onBookmark,
    onViewFull,
    onDownload,
    isExpanded,
    onToggleExpansion,
    downloadingCase,
  }) => {
    const getSourceInfo = (source) => {
      const sourceMap = {
        indian_kanoon: { name: "Indian Kanoon", icon: Scale, color: "blue", gradient: "from-blue-500 to-indigo-600" },
        supreme_court: { name: "Supreme Court", icon: Gavel, color: "red", gradient: "from-red-500 to-rose-600" },
        ecourts: { name: "eCourts", icon: Building, color: "green", gradient: "from-green-500 to-emerald-600" },
        legal_news: { name: "Legal News", icon: FileText, color: "purple", gradient: "from-purple-500 to-violet-600" },
      }
      return (
        sourceMap[source] || { name: source, icon: FileText, color: "gray", gradient: "from-gray-500 to-slate-600" }
      )
    }

    const sourceInfo = getSourceInfo(source)
    const Icon = sourceInfo.icon

    if (data.error) {
      return (
        <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-red-500">
          <div className="flex items-center mb-2">
            <Icon className={`h-6 w-6 mr-3 text-${sourceInfo.color}-600`} />
            <h3 className="text-lg font-semibold text-gray-900">{sourceInfo.name}</h3>
          </div>
          <p className="text-red-600">{data.error}</p>
        </div>
      )
    }

    // Handle Indian Kanoon results
    if (source === "indian_kanoon" && data.docs) {
      const displayCount = isExpanded ? data.docs.length : 5

      return (
        <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
          <div className={`bg-gradient-to-r ${sourceInfo.gradient} p-6 text-white`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Icon className="h-8 w-8 mr-4" />
                <div>
                  <h3 className="text-xl font-bold">{sourceInfo.name}</h3>
                  <p className="text-blue-100">Comprehensive legal database</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold">{data.found || data.docs.length}</div>
                <div className="text-sm text-blue-100">results found</div>
              </div>
            </div>
          </div>

          <div className="p-6">
            <div className="space-y-6">
              {data.docs.slice(0, displayCount).map((doc, index) => (
                <div
                  key={index}
                  className="group border border-gray-200 rounded-xl p-6 hover:shadow-lg hover:border-blue-300 transition-all duration-300 bg-gradient-to-r from-white to-gray-50"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h4 className="text-lg font-bold text-gray-900 group-hover:text-blue-600 transition-colors leading-tight flex-1 mr-4">
                      {doc.title}
                    </h4>
                    <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() =>
                          onBookmark({
                            title: doc.title,
                            source: sourceInfo.name,
                            tid: doc.tid,
                            headline: doc.headline,
                            docsource: doc.docsource,
                            citeList: doc.citeList,
                          })
                        }
                        className="p-2 text-gray-400 hover:text-yellow-500 hover:bg-yellow-50 rounded-lg transition-all"
                        title="Bookmark this case"
                      >
                        <Star className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => copyToClipboard(doc.title)}
                        className="p-2 text-gray-400 hover:text-blue-500 hover:bg-blue-50 rounded-lg transition-all"
                        title="Copy title"
                      >
                        <Copy className="h-5 w-5" />
                      </button>
                      <button
                        className="p-2 text-gray-400 hover:text-green-500 hover:bg-green-50 rounded-lg transition-all"
                        title="Share"
                      >
                        <Share2 className="h-5 w-5" />
                      </button>
                    </div>
                  </div>

                  <p className="text-gray-700 mb-4 leading-relaxed">{doc.headline}</p>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-6 text-sm text-gray-500">
                      <div className="flex items-center">
                        <Building className="h-4 w-4 mr-1" />
                        <span className="font-medium">{doc.docsource}</span>
                      </div>
                      {doc.docsize && (
                        <div className="flex items-center">
                          <FileText className="h-4 w-4 mr-1" />
                          <span>{doc.docsize} chars</span>
                        </div>
                      )}
                      {doc.citeList && (
                        <div className="flex items-center">
                          <BookOpen className="h-4 w-4 mr-1" />
                          <span>{doc.citeList.length} citations</span>
                        </div>
                      )}
                    </div>

                    <div className="flex space-x-3">
                      <button
                        onClick={() =>
                          onViewFull({
                            title: doc.title,
                            headline: doc.headline,
                            docsource: doc.docsource,
                            tid: doc.tid,
                            citeList: doc.citeList,
                          })
                        }
                        className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105"
                      >
                        <Eye className="h-4 w-4 mr-2" />
                        View Full
                      </button>
                      <button
                        onClick={() =>
                          onDownload({
                            title: doc.title,
                            headline: doc.headline,
                            docsource: doc.docsource,
                            tid: doc.tid,
                            citeList: doc.citeList,
                          })
                        }
                        disabled={downloadingCase === doc.tid}
                        className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105 disabled:opacity-50"
                      >
                        {downloadingCase === doc.tid ? (
                          <Loader className="animate-spin h-4 w-4 mr-2" />
                        ) : (
                          <Download className="h-4 w-4 mr-2" />
                        )}
                        Download
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {data.docs.length > 5 && (
              <div className="mt-6 text-center">
                <button
                  onClick={onToggleExpansion}
                  className="flex items-center mx-auto px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg font-medium transition-colors"
                >
                  {isExpanded ? (
                    <>
                      <ChevronUp className="h-5 w-5 mr-2" />
                      Show Less
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-5 w-5 mr-2" />
                      Show {data.docs.length - 5} More Results
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      )
    }

    // Handle other sources
    if (Array.isArray(data)) {
      return (
        <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
          <div className={`bg-gradient-to-r ${sourceInfo.gradient} p-6 text-white`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Icon className="h-8 w-8 mr-4" />
                <div>
                  <h3 className="text-xl font-bold">{sourceInfo.name}</h3>
                  <p className="text-blue-100">Legal information source</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold">{data.length}</div>
                <div className="text-sm text-blue-100">results</div>
              </div>
            </div>
          </div>

          <div className="p-6">
            <div className="space-y-4">
              {data.map((item, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-xl p-6 hover:shadow-md hover:border-blue-300 transition-all duration-200 bg-gradient-to-r from-white to-gray-50"
                >
                  <h4 className="font-bold text-gray-900 mb-3 text-lg">{item.title}</h4>
                  {item.snippet && <p className="text-gray-700 mb-4 leading-relaxed">{item.snippet}</p>}

                  <div className="flex items-center justify-between">
                    <div className="flex space-x-6 text-sm text-gray-500">
                      {item.court && (
                        <div className="flex items-center">
                          <Building className="h-4 w-4 mr-1" />
                          <span>{item.court}</span>
                        </div>
                      )}
                      {item.source && (
                        <div className="flex items-center">
                          <FileText className="h-4 w-4 mr-1" />
                          <span>{item.source}</span>
                        </div>
                      )}
                      {item.date && (
                        <div className="flex items-center">
                          <Calendar className="h-4 w-4 mr-1" />
                          <span>{item.date}</span>
                        </div>
                      )}
                    </div>

                    {item.link && (
                      <a
                        href={item.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105"
                      >
                        <ExternalLink className="h-4 w-4 mr-2" />
                        View Source
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    }

    return null
  }

  const AIAnalysisTab = () => {
    if (!aiAnalysis && !searchResults) {
      return (
        <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-2xl p-12 text-center">
          <div className="max-w-md mx-auto">
            <Zap className="h-16 w-16 text-purple-400 mx-auto mb-6" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">AI Legal Analysis</h3>
            <p className="text-gray-600 mb-6">
              Perform a search first, then click "AI Analysis" to get intelligent insights and comprehensive legal
              analysis.
            </p>
            <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center">
                <CheckCircle className="h-4 w-4 mr-1 text-green-500" />
                <span>Case Analysis</span>
              </div>
              <div className="flex items-center">
                <CheckCircle className="h-4 w-4 mr-1 text-green-500" />
                <span>Legal Precedents</span>
              </div>
              <div className="flex items-center">
                <CheckCircle className="h-4 w-4 mr-1 text-green-500" />
                <span>Key Insights</span>
              </div>
            </div>
          </div>
        </div>
      )
    }

    if (analyzingWithAI) {
      return (
        <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
          <div className="max-w-md mx-auto">
            <div className="relative mb-8">
              <Loader className="h-12 w-12 animate-spin text-blue-600 mx-auto" />
              <div className="absolute inset-0 rounded-full border-4 border-blue-200 animate-pulse"></div>
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-4">Analyzing Results...</h3>
            <p className="text-gray-600 mb-6">
              Our AI is analyzing the search results to provide comprehensive legal insights and identify key patterns.
            </p>
            <div className="flex justify-center space-x-2">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
            </div>
          </div>
        </div>
      )
    }

    if (aiAnalysis?.error) {
      return (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-xl p-8">
          <div className="flex items-center mb-4">
            <AlertCircle className="h-8 w-8 text-red-600 mr-4" />
            <div>
              <h3 className="text-xl font-bold text-red-800">Analysis Error</h3>
              <p className="text-red-600 mt-2">{aiAnalysis.error}</p>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-8 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Zap className="h-8 w-8 mr-4" />
              <div>
                <h3 className="text-2xl font-bold">AI Legal Analysis</h3>
                <p className="text-green-100">Comprehensive insights and legal interpretation</p>
              </div>
            </div>
            <div className="bg-white/20 backdrop-blur-sm px-4 py-2 rounded-full">
              <span className="text-sm font-medium">{aiAnalysis?.analysis_type || "Comprehensive"}</span>
            </div>
          </div>
        </div>

        <div className="p-8">
          <div className="prose max-w-none">
            <div className="bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl p-6 border-l-4 border-blue-500">
              <div className="whitespace-pre-wrap text-gray-800 leading-relaxed text-lg">
                {aiAnalysis?.analysis || "No analysis available"}
              </div>
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-200 flex items-center justify-between">
            <div className="text-sm text-gray-500">
              Analysis generated on {aiAnalysis?.timestamp && formatDate(aiAnalysis.timestamp)}
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => copyToClipboard(aiAnalysis?.analysis || "")}
                className="flex items-center px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
              >
                <Copy className="h-4 w-4 mr-2" />
                Copy Analysis
              </button>
              <button className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors">
                <Download className="h-4 w-4 mr-2" />
                Export PDF
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const LegalNewsTab = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-gray-900">Recent Legal News</h3>
          <p className="text-gray-600 mt-1">Stay updated with the latest legal developments</p>
        </div>
        <button
          onClick={fetchLegalNews}
          className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl font-medium transition-all duration-200 hover:scale-105"
        >
          <RefreshCw className="h-5 w-5 mr-2" />
          Refresh News
        </button>
      </div>

      {legalNews.length === 0 ? (
        <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-12 text-center">
          <FileText className="h-16 w-16 text-gray-400 mx-auto mb-6" />
          <h4 className="text-xl font-semibold text-gray-700 mb-2">No Recent News</h4>
          <p className="text-gray-600">Check back later for the latest legal news and updates.</p>
        </div>
      ) : (
        <div className="grid gap-6">
          {legalNews.map((news, index) => (
            <div
              key={index}
              className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-500 hover:shadow-xl transition-shadow duration-300"
            >
              <div className="flex justify-between items-start mb-4">
                <h4 className="text-xl font-bold text-gray-900 hover:text-purple-600 transition-colors flex-1 mr-4">
                  {news.title}
                </h4>
                <div className="flex items-center space-x-2">
                  <span className="bg-purple-100 text-purple-800 text-xs font-bold px-3 py-1 rounded-full">
                    {news.source}
                  </span>
                </div>
              </div>

              <p className="text-gray-700 mb-4 leading-relaxed">{news.snippet}</p>

              <div className="flex items-center justify-between">
                <div className="flex items-center text-sm text-gray-500">
                  <Calendar className="h-4 w-4 mr-1" />
                  <span>{news.date}</span>
                </div>
                {news.link && (
                  <a
                    href={news.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105"
                  >
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Read Full Article
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )

  const SavedSearchesTab = () => (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-2">Saved Searches & Bookmarks</h3>
        <p className="text-gray-600">Access your saved searches and bookmarked cases</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h4 className="text-xl font-bold text-gray-900 flex items-center">
              <Search className="h-6 w-6 mr-3 text-blue-600" />
              Recent Searches
            </h4>
            <span className="bg-blue-100 text-blue-800 text-sm font-bold px-3 py-1 rounded-full">
              {savedSearches.length}
            </span>
          </div>

          {savedSearches.length === 0 ? (
            <div className="text-center py-12">
              <Search className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No saved searches yet.</p>
              <p className="text-sm text-gray-400 mt-1">Your search history will appear here</p>
            </div>
          ) : (
            <div className="space-y-4">
              {savedSearches.map((search) => (
                <div
                  key={search.id}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md hover:border-blue-300 transition-all duration-200"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900 mb-1">{search.query}</p>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <div className="flex items-center">
                          <Clock className="h-4 w-4 mr-1" />
                          <span>{formatDate(search.timestamp)}</span>
                        </div>
                        <div className="flex items-center">
                          <FileText className="h-4 w-4 mr-1" />
                          <span>{search.resultCount} sources</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setSearchQuery(search.query)
                        setActiveTab("search")
                      }}
                      className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105"
                    >
                      <ArrowRight className="h-4 w-4 mr-1" />
                      Search Again
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h4 className="text-xl font-bold text-gray-900 flex items-center">
              <Star className="h-6 w-6 mr-3 text-yellow-600" />
              Bookmarked Cases
            </h4>
            <span className="bg-yellow-100 text-yellow-800 text-sm font-bold px-3 py-1 rounded-full">
              {bookmarkedCases.length}
            </span>
          </div>

          {bookmarkedCases.length === 0 ? (
            <div className="text-center py-12">
              <Bookmark className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No bookmarked cases yet.</p>
              <p className="text-sm text-gray-400 mt-1">Bookmark important cases for quick access</p>
            </div>
          ) : (
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {bookmarkedCases.slice(0, 10).map((bookmark) => (
                <div
                  key={bookmark.id}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md hover:border-yellow-300 transition-all duration-200"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900 mb-2">{bookmark.title}</p>
                      <p className="text-sm text-gray-600 mb-3 line-clamp-2">{bookmark.headline}</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span className="bg-gray-100 px-2 py-1 rounded">{bookmark.source}</span>
                        <div className="flex items-center">
                          <Clock className="h-3 w-3 mr-1" />
                          <span>Bookmarked {formatDate(bookmark.bookmarkedAt)}</span>
                        </div>
                      </div>
                    </div>
                    <Star className="h-5 w-5 text-yellow-500 ml-3 flex-shrink-0" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )

  const renderActiveTab = () => {
    switch (activeTab) {
      case "search":
        return <SearchResults />
      case "analysis":
        return <AIAnalysisTab />
      case "news":
        return <LegalNewsTab />
      case "saved":
        return <SavedSearchesTab />
      default:
        return <SearchResults />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <SearchHeader />
      <TabNavigation />

      <main className="max-w-7xl mx-auto px-6 py-8">{renderActiveTab()}</main>

      <CaseModal />

      {/* Enhanced Footer */}
      <footer className="bg-gradient-to-r from-gray-900 via-slate-800 to-gray-900 text-white py-12 mt-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div className="group">
              <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <Search className="h-8 w-8" />
              </div>
              <h4 className="text-3xl font-bold text-blue-400 mb-2">{savedSearches.length}</h4>
              <p className="text-gray-300">Saved Searches</p>
            </div>
            <div className="group">
              <div className="bg-yellow-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <Star className="h-8 w-8" />
              </div>
              <h4 className="text-3xl font-bold text-yellow-400 mb-2">{bookmarkedCases.length}</h4>
              <p className="text-gray-300">Bookmarked Cases</p>
            </div>
            <div className="group">
              <div className="bg-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <FileText className="h-8 w-8" />
              </div>
              <h4 className="text-3xl font-bold text-purple-400 mb-2">{legalNews.length}</h4>
              <p className="text-gray-300">Recent News</p>
            </div>
            <div className="group">
              <div className="bg-green-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <Award className="h-8 w-8" />
              </div>
              <h4 className="text-3xl font-bold text-green-400 mb-2">24/7</h4>
              <p className="text-gray-300">Available</p>
            </div>
          </div>

          <div className="mt-12 pt-8 border-t border-gray-700 text-center">
            <div className="flex items-center justify-center mb-4">
              <Scale className="h-8 w-8 text-amber-400 mr-3" />
              <span className="text-xl font-bold">Indian Legal Research Platform</span>
            </div>
            <p className="text-gray-300 max-w-2xl mx-auto">
              Empowering legal professionals with comprehensive research tools, AI-powered insights, and access to
              India's most extensive legal database.
            </p>
            <div className="mt-6 flex items-center justify-center space-x-6 text-sm text-gray-400">
              <span>© 2025 All rights reserved</span>
              <span>•</span>
              <span>Made with ❤️ for legal professionals</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default LegalResearchApp
