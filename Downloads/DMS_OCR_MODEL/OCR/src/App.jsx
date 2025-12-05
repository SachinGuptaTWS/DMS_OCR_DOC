import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, X, FileText, FileCode, FileSpreadsheet, Presentation,
  CheckCircle2, Settings, Play, BarChart3, Clock, Zap, 
  ArrowRight, Database, ServerCrash, Globe, Eye, Percent, CheckSquare,
  Network, User, Building, Cpu, Calendar, Table2, Image as ImageIcon, Layers,
  Grid, Layout, FileDigit, FolderOpen, FileCheck, FolderTree, HardDrive, Folder,
  ChevronRight, Search, Menu, Download, Filter, MapPin
} from 'lucide-react';

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
      <div className="bg-white border-l-4 border-blue-600 p-6 shadow-sm border-y border-r border-slate-200">
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
          <div key={category} className="bg-white border border-slate-200 flex flex-col h-full">
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
          <div className="px-4 py-3 border-b border-slate-200 flex justify-between items-center bg-slate-50">
            <h4 className="font-bold text-slate-700 text-xs uppercase tracking-wider flex items-center gap-2">
              Staged Files ({files.length})
            </h4>
            <button onClick={onNext} className="bg-slate-900 hover:bg-slate-800 text-white px-4 py-1.5 text-xs font-semibold tracking-wide transition-colors flex items-center gap-2 rounded-sm">
              CONFIGURE PIPELINE <ArrowRight className="w-3 h-3" />
            </button>
          </div>
          <div className="overflow-y-auto">
            <table className="min-w-full text-left text-xs">
                <thead className="bg-white border-b border-slate-100 text-slate-500">
                    <tr>
                        <th className="px-4 py-2 font-medium">Type</th>
                        <th className="px-4 py-2 font-medium">Filename</th>
                        <th className="px-4 py-2 font-medium">Size</th>
                        <th className="px-4 py-2 font-medium text-right">Action</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                    {files.map((file, i) => {
                        const { icon: Icon, color } = getFileIcon(file.name);
                        return (
                            <tr key={i} className="hover:bg-slate-50 group">
                                <td className="px-4 py-2 w-10">
                                    <Icon className={`w-4 h-4 ${color.split(' ')[0]}`} />
                                </td>
                                <td className="px-4 py-2 font-medium text-slate-700">{file.name}</td>
                                <td className="px-4 py-2 text-slate-500 font-mono">{formatBytes(file.size)}</td>
                                <td className="px-4 py-2 text-right">
                                    <button onClick={(e) => { e.stopPropagation(); removeFile(i); }} className="text-slate-400 hover:text-red-600 transition-colors">
                                        <X className="w-3.5 h-3.5" />
                                    </button>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function ConfigView({ files, onRun, isRunning }) {
  const groups = files.reduce((acc, f) => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    acc[ext] = (acc[ext] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        {/* Left Column: Summary */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div className="bg-white border border-slate-200 h-full flex flex-col">
            <div className="px-4 py-3 border-b border-slate-200 bg-slate-50">
                <h3 className="font-bold text-slate-700 text-xs uppercase tracking-wider flex items-center gap-2">
                <Database className="w-4 h-4 text-slate-500" />
                Manifest Summary
                </h3>
            </div>
            
            <div className="p-4 flex-1 overflow-y-auto">
                {Object.keys(groups).length === 0 ? (
                <p className="text-slate-400 italic text-xs text-center mt-10">No artifacts selected.</p>
                ) : (
                <div className="space-y-2">
                    {Object.entries(groups).map(([ext, count]) => {
                    const meta = STRATEGY_MAP[ext];
                    const { icon: Icon, color, bg, border } = getFileIcon(ext); 
                    return (
                        <div key={ext} className="flex items-center justify-between p-2 border border-slate-100 bg-slate-50">
                        <div className="flex items-center gap-3">
                            <div className={`p-1.5 border ${bg} ${color} ${border}`}>
                            <Icon className="w-3.5 h-3.5" />
                            </div>
                            <div>
                            <p className="text-xs font-bold text-slate-800">{ext.toUpperCase()}</p>
                            <p className="text-[10px] text-slate-500 uppercase tracking-tight">{meta?.cat || 'Unknown'}</p>
                            </div>
                        </div>
                        <span className="text-xs font-mono font-bold text-slate-600 px-2 py-0.5 bg-white border border-slate-200">{count}</span>
                        </div>
                    )
                    })}
                </div>
                )}
            </div>
          </div>
        </div>

        {/* Right Column: Engine Details */}
        <div className="lg:col-span-2 flex flex-col">
          <div className="bg-white border border-slate-200 flex-1 flex flex-col mb-6">
            <div className="px-4 py-3 border-b border-slate-200 bg-slate-50">
                <h3 className="font-bold text-slate-700 text-xs uppercase tracking-wider flex items-center gap-2">
                <Settings className="w-4 h-4 text-slate-500" />
                Execution Plan
                </h3>
            </div>
            <div className="p-4 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 gap-4">
              {Object.entries(STRATEGY_MAP).map(([ext, meta]) => {
                const isActive = groups[ext] > 0;
                if (!isActive && ['.jpg','.jpeg','.bmp','.tiff','.heic'].includes(ext)) return null;
                
                return (
                  <div key={ext} className={`p-3 border transition-all ${isActive ? 'border-blue-600 bg-blue-50/10' : 'border-slate-100 bg-slate-50 opacity-50 grayscale'}`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className={`font-mono text-[10px] font-bold px-1.5 py-0.5 border ${isActive ? 'bg-blue-100 text-blue-700 border-blue-200' : 'bg-slate-100 text-slate-500 border-slate-200'}`}>{ext}</span>
                      {isActive && <CheckSquare className="w-3.5 h-3.5 text-blue-600" />}
                    </div>
                    <div className="flex items-center gap-2 mb-1">
                        <p className="text-xs font-bold text-slate-800">{meta.tool}</p>
                    </div>
                    <p className="text-[10px] text-slate-500 border-t border-slate-100 pt-1 mt-1">{meta.strategy}</p>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="flex justify-end">
            <button 
              onClick={onRun} 
              disabled={isRunning}
              className={`
                px-6 py-3 font-semibold text-sm tracking-wide text-white transition-all rounded-sm flex items-center gap-2 shadow-sm
                ${isRunning ? 'bg-slate-400 cursor-not-allowed' : 'bg-blue-700 hover:bg-blue-800'}
              `}
            >
                {isRunning ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>EXECUTING BATCH JOB...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" />
                    <span>INITIALIZE EXTRACTION</span>
                  </>
                )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ResultsView({ results, onReset }) {
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);
  const selectedFile = results && results[selectedFileIdx];

  useEffect(() => {
    if (results?.length && !selectedFile) setSelectedFileIdx(0);
  }, [results]);

  if (!results) return null;

  return (
    <div className="h-full flex flex-col gap-6">
      
      {/* TOP METRICS */}
      {selectedFile && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard 
            label="Latency" 
            value={`${selectedFile.time_seconds ? selectedFile.time_seconds.toFixed(2) : '0.00'}s`} 
            icon={Clock} 
            colorClass="bg-blue-100 text-blue-700" 
          />
          <MetricCard 
            label="Confidence" 
            value={`${selectedFile.accuracy || 0}%`} 
            icon={Percent} 
            colorClass="bg-emerald-100 text-emerald-700" 
          />
          <MetricCard 
            label="Volume" 
            value={selectedFile.extracted_length ? selectedFile.extracted_length.toLocaleString() : "0"} 
            sub="Chars"
            icon={FileText} 
            colorClass="bg-slate-200 text-slate-700" 
          />
          <MetricCard 
            label="Entities" 
            value={selectedFile.ner_entities?.length || 0} 
            sub="Detected"
            icon={Network} 
            colorClass="bg-purple-100 text-purple-700" 
          />
        </div>
      )}

      {/* SPLIT VIEW */}
      <div className="flex-1 flex gap-4 overflow-hidden min-h-0">
        
        {/* Left: File List */}
        <div className="w-72 flex flex-col bg-white border border-slate-200">
          <div className="p-3 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
            <h4 className="font-bold text-slate-700 text-xs uppercase tracking-wider">Index</h4>
            <button onClick={onReset} className="text-[10px] text-blue-700 font-bold hover:underline uppercase">New Batch</button>
          </div>
          <div className="flex-1 overflow-y-auto">
            {results.map((res, i) => {
              const { icon: Icon, color } = getFileIcon(res.fileName);
              const isSelected = i === selectedFileIdx;
              return (
                <button 
                  key={i} 
                  onClick={() => setSelectedFileIdx(i)}
                  className={`w-full text-left px-4 py-3 border-b border-slate-100 transition-colors flex items-center justify-between group
                    ${isSelected ? 'bg-slate-50 border-l-4 border-l-blue-700' : 'hover:bg-slate-50 border-l-4 border-l-transparent'}`}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <Icon className={`w-3.5 h-3.5 ${color.split(' ')[0]}`} />
                        <p className={`font-semibold text-xs truncate ${isSelected ? 'text-slate-900' : 'text-slate-600'}`}>{res.fileName}</p>
                    </div>
                    <p className="text-[10px] text-slate-400 font-mono truncate pl-5.5">{res.engine}</p>
                  </div>
                  <div className={`w-2 h-2 rounded-full ${res.status === 'Success' ? 'bg-emerald-500' : 'bg-red-500'}`} />
                </button>
              )
            })}
          </div>
        </div>

        {/* Right: Detailed Viewer */}
        <div className="flex-1 bg-white border border-slate-200 flex flex-col">
          {selectedFile ? (
            <DetailedFileInspector result={selectedFile} />
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">Select an artifact to inspect details</div>
          )}
        </div>

      </div>
    </div>
  );
}

function DetailedFileInspector({ result }) {
  const [activeView, setActiveView] = useState('rich'); 
  const [compareText, setCompareText] = useState("");
  const [similarity, setSimilarity] = useState(null);

  const isSlide = result.fileName?.toLowerCase().endsWith('.pptx');
  const sectionLabel = isSlide ? "Slide" : "Page";

  const visualAssets = result.slides_data?.flatMap(s => 
    (s.items || []).filter(i => i.type === 'table_structure' || i.type === 'ocr_image').map(img => ({...img, slide: s.slide_number}))
  ) || [];

  const runComparison = () => {
    if (!compareText || !result.text_preview) return;
    const s1 = compareText.toLowerCase().replace(/\s+/g, '');
    const s2 = result.text_preview.toLowerCase().replace(/\s+/g, '');
    const len = Math.max(s1.length, s2.length);
    if (len === 0) return;
    let matches = 0;
    for(let i=0; i<Math.min(s1.length, s2.length); i++) if(s1[i]===s2[i]) matches++;
    setSimilarity(((matches/len)*100).toFixed(1));
  };

  const tabs = [
    { id: 'rich', label: 'Structured View', icon: Layout },
    { id: 'raw', label: 'Raw Text', icon: FileText },
    { id: 'smart', label: `Entities (${result.ner_entities?.length || 0})`, icon: Network },
    { id: 'visual', label: `Assets (${visualAssets.length})`, icon: ImageIcon }
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Tabs */}
      <div className="flex items-center border-b border-slate-200 bg-slate-50 px-4">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveView(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-3 text-xs font-bold uppercase tracking-wide border-b-2 transition-colors
              ${activeView === tab.id ? 'border-blue-600 text-blue-700 bg-white' : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'}
            `}
          >
            <tab.icon className="w-3.5 h-3.5" /> {tab.label}
          </button>
        ))}
      </div>

      {/* Viewport */}
      <div className="flex-1 overflow-y-auto p-6 bg-white">
        
        {/* 1. RICH VIEW */}
        {activeView === 'rich' && (
          <div className="space-y-6 max-w-5xl mx-auto">
            {result.slides_data && result.slides_data.length > 0 ? (
              result.slides_data.map((slide, i) => (
                <div key={i} className="border border-slate-200">
                  <div className="flex items-center justify-between px-4 py-2 bg-slate-50 border-b border-slate-200">
                    <h4 className="font-bold text-slate-700 text-xs uppercase flex items-center gap-2">
                      <Layers className="w-3.5 h-3.5 text-slate-500" />
                      {sectionLabel} {slide.slide_number}
                    </h4>
                    <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 border border-emerald-100 px-2 py-0.5 rounded">Conf: {slide.accuracy_score}%</span>
                  </div>
                  <div className="p-6">
                    {slide.items && slide.items.map((item, j) => (
                      <div key={j} className="mb-4 last:mb-0">
                         <SlideItemRenderer item={item} />
                      </div>
                    ))}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-20 bg-slate-50 border border-dashed border-slate-300">
                <p className="text-slate-500 text-sm mb-4">No structured hierarchy detected.</p>
                <button onClick={() => setActiveView('raw')} className="text-blue-700 text-xs font-bold hover:underline">SWITCH TO RAW TEXT</button>
              </div>
            )}
          </div>
        )}

        {/* 2. RAW TEXT */}
        {activeView === 'raw' && (
          <div className="grid grid-cols-2 gap-6 h-full">
            <div className="flex flex-col h-full">
              <span className="text-xs font-bold text-slate-500 uppercase mb-2 block">System Output</span>
              <textarea 
                readOnly 
                className="flex-1 w-full p-4 font-mono text-xs border border-slate-200 bg-slate-50 text-slate-700 resize-none focus:outline-none"
                value={result.text_preview || "No content extracted"}
              />
            </div>
            <div className="flex flex-col h-full">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-slate-500 uppercase">Ground Truth / Diff</span>
                {similarity && <Badge variant="brand">MATCH: {similarity}%</Badge>}
              </div>
              <textarea 
                placeholder="Paste original text here to compare accuracy..." 
                className="flex-1 w-full p-4 font-mono text-xs border border-slate-200 bg-white text-slate-700 resize-none focus:outline-none focus:border-blue-500 transition-colors"
                value={compareText}
                onChange={(e) => setCompareText(e.target.value)}
              />
              <button onClick={runComparison} className="mt-2 w-full bg-slate-800 text-white py-2 text-xs font-bold uppercase tracking-wider hover:bg-slate-900">Calculate Similarity</button>
            </div>
          </div>
        )}

        {/* 3. VISUAL ASSETS */}
        {activeView === 'visual' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {visualAssets.length > 0 ? visualAssets.map((asset, i) => (
              <div key={i} className="border border-slate-200">
                <div className="bg-slate-50 p-2 border-b border-slate-200 flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    {asset.type === 'table_structure' ? <Table2 className="w-3.5 h-3.5 text-emerald-600"/> : <ImageIcon className="w-3.5 h-3.5 text-purple-600"/>}
                    <span className="text-[10px] font-bold text-slate-700 uppercase">{asset.type === 'table_structure' ? 'Table Data' : 'Image Asset'}</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">#{i+1}</span>
                </div>
                <div className="p-4 overflow-auto max-h-60 text-xs">
                  <SlideItemRenderer item={asset} />
                </div>
              </div>
            )) : (
              <div className="col-span-2 text-center py-20 text-slate-400 border border-dashed border-slate-300 bg-slate-50">
                <p className="text-sm">No visual assets extracted.</p>
              </div>
            )}
          </div>
        )}

        {/* 4. INTELLIGENCE (NER) */}
        {activeView === 'smart' && (
          <div className="border border-slate-200 p-6 min-h-full">
            <h4 className="font-bold text-slate-800 mb-6 flex items-center gap-2 text-sm uppercase tracking-wide">
              <Network className="w-4 h-4 text-slate-400" />
              Named Entity Recognition (GLiNER)
            </h4>
            
            {result.ner_entities && result.ner_entities.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {result.ner_entities.map((e, i) => <EntityChip key={i} entity={e} />)}
              </div>
            ) : (
              <p className="text-slate-400 text-sm italic border-l-2 border-slate-200 pl-3">No entities detected or NER engine disabled.</p>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

// --- MAIN LAYOUT ---

const NavButton = ({ icon: Icon, label, active, onClick, disabled, badge }) => (
  <button 
    onClick={disabled ? undefined : onClick}
    className={`
      w-full flex items-center justify-between px-4 py-3 text-sm font-medium transition-colors border-l-4
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
    <div className="flex h-screen bg-slate-100 font-sans text-slate-900 overflow-hidden antialiased">
      
      {/* SIDEBAR */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col z-20 shadow-xl border-r border-slate-800">
        <div className="h-16 flex items-center px-6 border-b border-slate-800 bg-slate-950">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-700 flex items-center justify-center rounded-sm">
              <Zap className="w-5 h-5 text-white" fill="white" />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-wide text-white">OCR CORE</h1>
              <p className="text-[10px] text-slate-400 uppercase tracking-widest">Enterprise v3.5</p>
            </div>
          </div>
        </div>

        <div className="flex-1 py-6 space-y-1">
          <p className="px-6 text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2">Modules</p>
          <NavButton icon={Upload} label="Ingestion" active={activeTab === 'upload'} onClick={() => setActiveTab('upload')} badge={files.length || null} />
          <NavButton icon={Settings} label="Pipeline Config" active={activeTab === 'config'} onClick={() => setActiveTab('config')} disabled={files.length === 0} />
          <NavButton icon={BarChart3} label="Reporting" active={activeTab === 'results'} onClick={() => setActiveTab('results')} disabled={!results} />
          <NavButton icon={FolderOpen} label="Smart Archive" active={activeTab === 'storage'} onClick={() => setActiveTab('storage')} disabled={!results} />
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-slate-100">
        
        {/* HEADER */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 shadow-sm z-10">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                {activeTab === 'upload' && 'Document Ingestion'}
                {activeTab === 'config' && 'Pipeline Configuration'}
                {activeTab === 'results' && 'Intelligence Report'}
                {activeTab === 'storage' && 'Taxonomy Organizer'}
            </h2>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input type="text" placeholder="Global Search..." className="pl-9 pr-4 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-sm focus:outline-none focus:border-blue-400 w-64 transition-colors" />
            </div>
            <div className="w-8 h-8 bg-slate-100 border border-slate-200 rounded-full flex items-center justify-center text-slate-600 font-bold text-xs">
                <User className="w-4 h-4" />
            </div>
          </div>
        </header>

        {error && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-50 bg-red-600 text-white px-6 py-3 rounded shadow-lg flex items-center gap-4 animate-in slide-in-from-top-2">
            <ServerCrash className="w-5 h-5" />
            <div>
              <p className="font-bold text-sm">System Error</p>
              <p className="text-xs opacity-90">{error}</p>
            </div>
            <button onClick={() => setError(null)} className="ml-4 hover:bg-red-700 p-1 rounded"><X className="w-4 h-4" /></button>
          </div>
        )}

        <div className="flex-1 overflow-auto p-8">
          <div className="max-w-7xl mx-auto h-full">
            {activeTab === 'upload' && <UploadView files={files} setFiles={setFiles} onNext={() => setActiveTab('config')} />}
            {activeTab === 'config' && <ConfigView files={files} onRun={handleRun} isRunning={isRunning} />}
            {activeTab === 'results' && <ResultsView results={results} onReset={() => { setFiles([]); setResults(null); setActiveTab('upload'); }} />}
            {activeTab === 'storage' && <SmartStorageView results={results} />}
          </div>
        </div>

      </main>
    </div>
  );
}