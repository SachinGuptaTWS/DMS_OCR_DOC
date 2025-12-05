import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, X, FileText, FileCode, FileSpreadsheet, Presentation,
  CheckCircle2, Settings, Play, BarChart3, Clock, Zap, 
  ArrowRight, Database, ServerCrash, Globe, Eye, Percent, CheckSquare,
  Network, User, Building, Cpu, Calendar, Table2, Image as ImageIcon, Layers,
  Grid, Layout, FileDigit, FolderOpen, FileCheck, FolderTree, HardDrive, Folder,
  ChevronRight, Search, Menu, Download, Filter, MapPin, Server
} from 'lucide-react';
import TestConnection from './components/TestConnection';

const API_ENDPOINT = "http://localhost:8000/api/benchmark";

// --- CONFIGURATION ---
const STRATEGY_MAP = {
  '.txt':  { cat: 'Text Script', strategy: 'Native IO', tool: '_txt_OCR.py' },
  '.csv':  { cat: 'Data Script', strategy: 'Pandas Intelligent', tool: '_csv_ocr.py' },
  '.xml':  { cat: 'XML Script', strategy: 'BS4/LXML', tool: '_xml_ocr.py' },
  '.html': { cat: 'HTML Script', strategy: 'BS4', tool: '_html_ocr.py' },
  '.docx': { cat: 'Word Document', strategy: 'Native (python-docx)', tool: '_docx_ocr.py' },
  '.pptx': { cat: 'Presentation', strategy: 'Hybrid (Gemini+EasyOCR)', tool: '_ppt_ocr.py' },
  '.pdf':  { cat: 'Document', strategy: 'Hybrid (Camelot+Paddle)', tool: '_pdf_ocr.py' },
  '.png':  { cat: 'Image', strategy: 'Hybrid Vision (Gemini+Paddle)', tool: '_image_ocr.py' },
  '.jpg':  { cat: 'Image', strategy: 'Hybrid Vision (Gemini+Paddle)', tool: '_image_ocr.py' },
  '.jpeg': { cat: 'Image', strategy: 'Hybrid Vision (Gemini+Paddle)', tool: '_image_ocr.py' },
  '.bmp':  { cat: 'Image', strategy: 'Hybrid Vision (Gemini+Paddle)', tool: '_image_ocr.py' },
  '.tiff': { cat: 'Image', strategy: 'Hybrid Vision (Gemini+Paddle)', tool: '_image_ocr.py' },
  '.heic': { cat: 'Image', strategy: 'Hybrid Vision (Gemini+Paddle)', tool: '_image_ocr.py' }
};

const ALLOWED_EXTENSIONS = Object.keys(STRATEGY_MAP);
const ACCEPT_STRING = ALLOWED_EXTENSIONS.join(',');

// --- UTILS ---
const getFileIcon = (fileName) => {
  if (!fileName) return { icon: FileText, color: 'text-slate-500', bg: 'bg-slate-100', border: 'border-slate-300' };
  const ext = '.' + fileName.split('.').pop().toLowerCase();
  
  if (ext === '.pptx') return { icon: Presentation, color: 'text-orange-700', bg: 'bg-orange-50', border: 'border-orange-200' };
  if (ext === '.csv') return { icon: FileSpreadsheet, color: 'text-emerald-700', bg: 'bg-emerald-50', border: 'border-emerald-200' };
  if (ext === '.xml') return { icon: FileCode, color: 'text-blue-700', bg: 'bg-blue-50', border: 'border-blue-200' };
  if (ext === '.docx') return { icon: FileText, color: 'text-blue-700', bg: 'bg-blue-50', border: 'border-blue-200' };
  if (ext === '.html') return { icon: Globe, color: 'text-indigo-700', bg: 'bg-indigo-50', border: 'border-indigo-200' };
  if (ext === '.pdf') return { icon: FileDigit, color: 'text-red-700', bg: 'bg-red-50', border: 'border-red-200' };
  
  if (['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.heic'].includes(ext)) {
    return { icon: ImageIcon, color: 'text-purple-700', bg: 'bg-purple-50', border: 'border-purple-200' };
  }

  return { icon: FileText, color: 'text-slate-600', bg: 'bg-slate-100', border: 'border-slate-300' };
};

const formatBytes = (bytes) => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

// --- COMPONENTS ---

const Badge = ({ children, variant = 'neutral' }) => {
  const styles = {
    neutral: 'bg-slate-100 text-slate-600 border-slate-200',
    success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    error: 'bg-red-50 text-red-700 border-red-200',
    brand: 'bg-blue-50 text-blue-700 border-blue-200',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${styles[variant]} uppercase tracking-wide`}>
      {children}
    </span>
  );
};

const MetricCard = ({ label, value, sub, icon: Icon, colorClass }) => (
  <div className="bg-white p-4 border border-slate-200 flex items-start justify-between">
    <div>
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">{label}</p>
      <div className="flex items-baseline gap-2">
        <h3 className="text-2xl font-bold text-slate-900 leading-none">
            {typeof value === 'object' ? JSON.stringify(value) : value}
        </h3>
        {sub && <span className="text-xs font-medium text-slate-400">{sub}</span>}
      </div>
    </div>
    <div className={`p-2 rounded ${colorClass} bg-opacity-10`}>
      {Icon && <Icon className={`w-5 h-5 ${colorClass.replace('bg-', 'text-')}`} />}
    </div>
  </div>
);

const EntityChip = ({ entity }) => {
  if (!entity || !entity.text) return null;

  let color = "bg-slate-100 text-slate-700 border-slate-300";
  let Icon = Database;

  const l = (entity.label || "").toLowerCase();
  if (l.includes('person')) { color = "bg-purple-100 text-purple-800 border-purple-200"; Icon = User; }
  else if (l.includes('org') || l.includes('company')) { color = "bg-blue-100 text-blue-800 border-blue-200"; Icon = Building; }
  else if (l.includes('tech') || l.includes('soft')) { color = "bg-cyan-100 text-cyan-800 border-cyan-200"; Icon = Cpu; }
  else if (l.includes('date') || l.includes('time')) { color = "bg-orange-100 text-orange-800 border-orange-200"; Icon = Calendar; }
  else if (l.includes('money') || l.includes('curr')) { color = "bg-emerald-100 text-emerald-800 border-emerald-200"; Icon = CheckCircle2; }

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium border ${color}`}>
      <Icon className="w-3 h-3 opacity-70" />
      <span className="truncate max-w-[180px]" title={entity.text}>
        {typeof entity.text === 'string' ? entity.text : JSON.stringify(entity.text)}
      </span>
      <span className="opacity-60 text-[9px] uppercase border-l border-current pl-1 ml-0.5">{entity.label}</span>
    </span>
  );
};

const MarkdownTable = ({ content }) => {
  if (!content) return null;
  const safeContent = typeof content === 'string' ? content : JSON.stringify(content, null, 2);

  try {
    const rows = safeContent.trim().split('\n').map(r => {
        const cells = r.split('|');
        if (cells.length > 2) return cells.slice(1, -1).map(c => c.trim());
        return [];
    }).filter(r => r.length > 0);

    if (rows.length < 2) return <pre className="text-xs bg-slate-50 p-4 border border-slate-200 font-mono overflow-x-auto text-slate-700">{safeContent}</pre>;

    const header = rows[0];
    const body = rows.slice(2);

    return (
      <div className="overflow-x-auto my-4 border border-slate-200 bg-white">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50">
            <tr>
              {header.map((h, i) => (
                <th key={i} className="px-4 py-2 text-left font-bold text-slate-700 text-xs uppercase tracking-wider border-r border-slate-200 last:border-0 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {body.map((row, i) => (
              <tr key={i} className="hover:bg-slate-50 transition-colors">
                {row.map((cell, j) => (
                  <td key={j} className="px-4 py-2 text-slate-700 border-r border-slate-100 last:border-0 text-xs">{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  } catch (e) {
    return <div className="text-xs text-red-600 bg-red-50 p-2 border border-red-200">Table Rendering Error</div>;
  }
};

const SlideItemRenderer = ({ item }) => {
  if (!item) return null;

  if (item.type === 'table_structure') {
    return (
      <div className="mb-6 border border-slate-200 bg-white p-4">
        <div className="flex items-center gap-2 text-xs font-bold text-slate-500 mb-3 uppercase tracking-wide border-b border-slate-100 pb-2">
          <Table2 className="w-3.5 h-3.5" /> Structured Table Data
        </div>
        <MarkdownTable content={item.text} />
      </div>
    );
  }
  
  if (item.type === 'ocr_image') {
    const rawText = typeof item.text === 'string' ? item.text : JSON.stringify(item.text || "");
    const cleanText = rawText.replace('[IMAGE TEXT]:', '').replace('[GEMINI VISION]:', '').trim() || "No text extracted";
    
    return (
      <div className="mb-4 p-4 bg-slate-50 border border-slate-200">
        <div className="flex items-center justify-between mb-2">
           <div className="flex items-center gap-2 text-xs font-bold text-slate-600 uppercase tracking-wide">
             <ImageIcon className="w-3.5 h-3.5" /> Visual Context Analysis
           </div>
           <Badge variant="brand">Confidence: {item.conf ? (item.conf * 100).toFixed(0) : 0}%</Badge>
        </div>
        <p className="text-sm text-slate-700 leading-relaxed font-mono bg-white p-3 border border-slate-100">{cleanText}</p>
      </div>
    );
  }

  const displayContent = typeof item.text === 'string' ? item.text : (typeof item === 'string' ? item : JSON.stringify(item));

  return (
    <div className="mb-3 text-sm text-slate-800 leading-relaxed whitespace-pre-wrap font-sans border-l-2 border-slate-100 pl-3">
      {displayContent}
    </div>
  );
};

// --- SUB-VIEWS ---

function SmartStorageView({ results }) {
  const folders = (results || []).reduce((acc, file) => {
    const cat = file.doc_category || "Uncategorized";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(file);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <div className="relative bg-white p-6 shadow-sm border border-slate-200">
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-600"></div>
        <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
          <FolderTree className="w-5 h-5 text-blue-600" />
          Intelligent Corporate Taxonomy
        </h3>
        <p className="text-slate-500 mt-1 text-sm">
          Gemini Flash 1.5 has analyzed, classified, and physically moved your files into the following directory structure.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(folders).map(([category, files]) => (
          <div key={category} className="bg-white border-x border-b border-slate-200 flex flex-col h-full">
            <div className="bg-slate-50 px-4 py-3 border-b border-slate-200 flex items-center justify-between">
               <div className="flex items-center gap-2">
                 <Folder className="w-4 h-4 text-slate-500 fill-blue-100" />
                 <span className="font-bold text-slate-800 text-sm uppercase tracking-wide">{category}</span>
               </div>
               <span className="bg-white border border-slate-200 text-slate-500 text-[10px] font-bold px-2 py-0.5 rounded shadow-sm">{files.length}</span>
            </div>
            <div className="p-0 flex-1">
              {files.map((file, i) => (
                <div key={i} className="group px-4 py-3 border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors cursor-default">
                  <div className="flex items-start gap-3">
                    <FileCheck className="w-4 h-4 text-emerald-600 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-700 truncate">{file.fileName}</p>
                      <div className="flex items-center gap-1 mt-1 text-[10px] text-slate-400 font-mono">
                        <MapPin className="w-3 h-3" />
                        <span className="truncate">{file.storage_path || "Processing..."}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function UploadView({ files, setFiles, onNext }) {
  const inputRef = useRef(null);
  
  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files?.length) {
      const newFiles = Array.from(e.dataTransfer.files).filter(f => {
         const ext = '.' + f.name.split('.').pop().toLowerCase();
         return ALLOWED_EXTENSIONS.includes(ext);
      });
      if (newFiles.length > 0) setFiles(prev => [...prev, ...newFiles]);
    }
  };

  const removeFile = (idx) => setFiles(p => p.filter((_, i) => i !== idx));

  return (
    <div className="h-full flex flex-col gap-6">
      <div 
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="flex-1 min-h-[300px] border border-slate-300 bg-slate-50 hover:bg-slate-100 hover:border-slate-400 transition-all cursor-pointer flex flex-col items-center justify-center p-10 group relative"
        style={{ backgroundImage: 'radial-gradient(#cbd5e1 1px, transparent 1px)', backgroundSize: '20px 20px' }}
      >
        <input type="file" multiple className="hidden" ref={inputRef} accept={ACCEPT_STRING} onChange={(e) => e.target.files?.length && setFiles(p => [...p, ...Array.from(e.target.files)])} />
        
        <div className="w-16 h-16 bg-white border border-slate-200 text-slate-400 rounded-sm flex items-center justify-center mb-4 shadow-sm group-hover:text-blue-600 group-hover:border-blue-200 transition-colors z-10">
          <Upload className="w-8 h-8" strokeWidth={1} />
        </div>
        <h3 className="text-lg font-bold text-slate-800 mb-1 z-10">Upload Data Artifacts</h3>
        <p className="text-slate-500 text-sm text-center max-w-sm mb-6 z-10">
          Drag and drop compatible files (PDF, CSV, Office, Images) to begin the ingestion pipeline.
        </p>
      </div>

      {files.length > 0 && (
        <div className="bg-white border border-slate-200 flex flex-col max-h-[300px]">
          <div className="px-4 py-3 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
            <h4 className="font-bold text-slate-700 text-xs uppercase tracking-wider">Staged Files ({files.length})</h4>
            <button onClick={onNext} className="bg-slate-900 hover:bg-slate-800 text-white px-4 py-1.5 text-xs font-semibold tracking-wide transition-colors flex items-center gap-2 rounded-sm">
              CONFIGURE PIPELINE <ArrowRight className="w-3 h-3" />
            </button>
          </div>
          <div className="overflow-y-auto">
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">File</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Size</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">Action</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-slate-200">
                {files.map((file, index) => (
                  <tr key={index}>
                    <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-slate-900">{file.name}</td>
                    <td className="px-4 py-2 whitespace-nowrap text-sm text-slate-500">{(file.size / 1024).toFixed(1)} KB</td>
                    <td className="px-4 py-2 whitespace-nowrap text-right text-sm font-medium">
                      <button 
                        onClick={() => {
                          const newFiles = [...files];
                          newFiles.splice(index, 1);
                          setFiles(newFiles);
                        }}
                        className="text-red-600 hover:text-red-900"
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// --- MAIN LAYOUT ---

const NavButton = ({ icon: Icon, label, active, onClick, disabled, badge }) => (
  <button 
    onClick={disabled ? undefined : onClick}
    className={`w-full flex items-center justify-between px-4 py-3 text-sm font-medium transition-colors border-l-4
      ${active 
        ? 'bg-slate-800 border-l-blue-500 text-white' 
        : disabled 
          ? 'text-slate-600 cursor-not-allowed opacity-50 border-l-transparent' 
          : 'text-slate-400 hover:bg-slate-800 hover:text-white border-l-transparent'
      }
    `}
  >
    <div className="flex items-center gap-3">
      {Icon && <Icon className={`w-4 h-4 ${active ? 'text-blue-400' : ''}`} />}
      {label}
    </div>
    {badge && <span className="bg-blue-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-sm">{badge}</span>}
  </button>
);

export default function OCRBenchmarkApp() {
  const [activeTab, setActiveTab] = useState('upload');
  const [files, setFiles] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true);

  const handleRun = async () => {
    if (files.length === 0) return;
    setIsRunning(true);
    setError(null);
    
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('config', JSON.stringify({ mode: 'auto_script_dispatch' }));

    try {
      const res = await fetch(API_ENDPOINT, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`API Error: ${res.status} ${res.statusText}`);
      const data = await res.json();
      
      setResults(data.results || []);
      setActiveTab('results');
    } catch (err) {
      setError(err.message || "Failed to connect to backend server.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <Cpu className="h-8 w-8 text-blue-600" />
                <span className="ml-2 text-xl font-semibold text-gray-900">OCR Benchmark</span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 text-sm text-gray-500">
                <Server className="h-4 w-4 text-green-500" />
                <span>Backend: {process.env.VITE_API_BASE_URL || 'Not configured'}</span>
              </div>
              <button
                onClick={() => setShowSidebar(!showSidebar)}
                className="md:hidden p-2 rounded-md text-gray-500 hover:bg-gray-100 focus:outline-none"
              >
                <Menu className="h-6 w-6" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className={`${showSidebar ? 'w-64' : 'w-0'} bg-slate-50 border-r border-slate-200 overflow-hidden transition-all duration-200`}>
          <div className="h-full flex flex-col">
            <div className="p-4 border-b border-slate-200">
              <h2 className="text-lg font-semibold text-slate-800">Navigation</h2>
            </div>
            <nav className="flex-1 overflow-y-auto p-2 space-y-1">
              <NavButton 
                icon={Upload} 
                label="Upload Files" 
                active={activeTab === 'upload'} 
                onClick={() => setActiveTab('upload')} 
              />
              <NavButton 
                icon={Settings} 
                label="Configuration" 
                active={activeTab === 'config'} 
                onClick={() => setActiveTab('config')} 
                disabled={files.length === 0} 
              />
              <NavButton 
                icon={Database} 
                label="Results" 
                active={activeTab === 'results'} 
                onClick={() => setActiveTab('results')} 
                disabled={!results || results.length === 0} 
              />
              <NavButton 
                icon={Folder} 
                label="Storage" 
                active={activeTab === 'storage'} 
                onClick={() => setActiveTab('storage')} 
              />
            </nav>
            <div className="p-4 border-t border-slate-200">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Server className="w-4 h-4" />
                <span>Backend: {process.env.VITE_API_BASE_URL ? 'Connected' : 'Not Configured'}</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Content Header */}
          <div className="bg-white border-b border-slate-200 p-4">
            <h2 className="text-lg font-semibold text-slate-800">
              {activeTab === 'upload' && 'Document Upload'}
              {activeTab === 'config' && 'Pipeline Configuration'}
              {activeTab === 'results' && 'Processing Results'}
              {activeTab === 'storage' && 'Smart Storage'}
            </h2>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-auto p-6 bg-slate-50">
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
                <div className="flex">
                  <div className="shrink-0">
                    <XCircleIcon className="h-5 w-5 text-red-500" />
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                  <div className="ml-auto pl-3">
                    <div className="-mx-1.5 -my-1.5">
                      <button
                        type="button"
                        onClick={() => setError(null)}
                        className="inline-flex rounded-md bg-red-50 p-1.5 text-red-500 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-600 focus:ring-offset-2 focus:ring-offset-red-50"
                      >
                        <span className="sr-only">Dismiss</span>
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="h-full">
              {activeTab === 'upload' && <UploadView files={files} setFiles={setFiles} onNext={() => setActiveTab('config')} />}
              {activeTab === 'config' && <ConfigView files={files} onRun={handleRun} isRunning={isRunning} />}
              {activeTab === 'results' && <ResultsView results={results} onReset={() => { setFiles([]); setResults(null); setActiveTab('upload'); }} />}
              {activeTab === 'storage' && <SmartStorageView results={results} />}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}