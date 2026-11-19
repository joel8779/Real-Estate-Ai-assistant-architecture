import React from 'react';
import { Building2, RotateCcw } from 'lucide-react';

interface NavbarProps {
  sourceFilter: 'all' | 'darglobal' | 'wasalt';
  setSourceFilter: (filter: 'all' | 'darglobal' | 'wasalt') => void;
  isBackendHealthy: boolean | null;
  onClearChat?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  sourceFilter,
  setSourceFilter,
  isBackendHealthy,
  onClearChat,
}) => {
  return (
    <header className="border-b border-surface-border bg-surface-base/90 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-surface-subtle border border-surface-border flex items-center justify-center">
          <Building2 className="w-4 h-4 text-luxury-400" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-sm font-semibold tracking-tight text-white">
              DarGlobal <span className="text-[#A1A1AA] font-normal">&</span> Wasalt
            </h1>
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-surface-subtle border border-surface-border text-[#A1A1AA] font-medium">
              Assistant
            </span>
          </div>
          <p className="text-[11px] text-[#A1A1AA] hidden sm:block">
            Grounded real estate intelligence across global developments & domestic listings
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-2 sm:space-x-3">
        {/* Source Filter Switcher */}
        <div className="flex bg-surface-card p-0.5 rounded-lg border border-surface-border text-xs">
          <button
            type="button"
            onClick={() => setSourceFilter('all')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              sourceFilter === 'all'
                ? 'bg-surface-subtle text-white font-medium shadow-sm'
                : 'text-[#A1A1AA] hover:text-white'
            }`}
          >
            All Sources
          </button>
          <button
            type="button"
            onClick={() => setSourceFilter('darglobal')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              sourceFilter === 'darglobal'
                ? 'bg-luxury-500/20 text-luxury-300 font-medium border border-luxury-500/30'
                : 'text-[#A1A1AA] hover:text-white'
            }`}
          >
            DarGlobal
          </button>
          <button
            type="button"
            onClick={() => setSourceFilter('wasalt')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              sourceFilter === 'wasalt'
                ? 'bg-emerald-500/20 text-emerald-300 font-medium border border-emerald-500/30'
                : 'text-[#A1A1AA] hover:text-white'
            }`}
          >
            Wasalt
          </button>
        </div>

        {/* Clear Chat Action */}
        {onClearChat && (
          <button
            type="button"
            onClick={onClearChat}
            className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-surface-card hover:bg-surface-subtle border border-surface-border text-[#A1A1AA] hover:text-white text-xs transition-all"
            title="Start fresh conversation"
          >
            <RotateCcw className="w-3.5 h-3.5 text-[#A1A1AA]" />
            <span className="hidden sm:inline text-[11px]">Clear</span>
          </button>
        )}

        {/* Backend Connectivity Status */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-surface-card border border-surface-border text-xs">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isBackendHealthy === true
                ? 'bg-emerald-400'
                : isBackendHealthy === false
                ? 'bg-red-400'
                : 'bg-amber-400'
            }`}
          />
          <span className="text-[#A1A1AA] text-[11px] hidden md:inline">
            {isBackendHealthy === true
              ? 'Online'
              : isBackendHealthy === false
              ? 'Disconnected'
              : 'Checking...'}
          </span>
        </div>
      </div>
    </header>
  );
};
