import React from 'react';
import { ExternalLink } from 'lucide-react';
import { Citation } from '../types';

interface CitationCardProps {
  citation: Citation;
}

export const CitationCard: React.FC<CitationCardProps> = ({ citation }) => {
  const isDarGlobal = citation.source === 'darglobal';

  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-start justify-between gap-3 p-2.5 rounded-lg bg-surface-card hover:bg-surface-subtle border border-surface-border hover:border-[#3F3F46] transition-all text-left"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span
            className={`text-[10px] font-medium tracking-wide uppercase px-1.5 py-0.5 rounded ${
              isDarGlobal
                ? 'bg-luxury-500/10 text-luxury-300 border border-luxury-500/20'
                : 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
            }`}
          >
            {isDarGlobal ? 'DarGlobal' : 'Wasalt'}
          </span>
          <span className="text-[10px] text-[#A1A1AA] capitalize">
            {(citation.category || 'project').replace(/_/g, ' ')}
          </span>
          {citation.provenance === 'demonstration_unverified' && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Demo
            </span>
          )}
        </div>
        <h5 className="text-xs font-medium text-[#F5F5F5] group-hover:text-white truncate">
          {citation.title}
        </h5>
        {citation.snippet && (
          <p className="text-[11px] text-[#A1A1AA] line-clamp-1 mt-0.5 leading-relaxed">
            {citation.snippet}
          </p>
        )}
      </div>
      <ExternalLink className="w-3.5 h-3.5 text-[#71717A] group-hover:text-[#F5F5F5] flex-shrink-0 mt-0.5 transition-colors" />
    </a>
  );
};

