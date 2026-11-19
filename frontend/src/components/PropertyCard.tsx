import React from 'react';
import { MapPin, BedDouble, ExternalLink } from 'lucide-react';
import { PropertyCardData } from '../types';

interface PropertyCardProps {
  property: PropertyCardData;
}

export const PropertyCard: React.FC<PropertyCardProps> = ({ property }) => {
  return (
    <div className="w-64 flex-shrink-0 rounded-xl bg-surface-card border border-surface-border overflow-hidden hover:border-[#3F3F46] transition-all flex flex-col justify-between">
      {property.image_url ? (
        <div className="h-32 w-full overflow-hidden relative bg-surface-subtle">
          <img
            src={property.image_url}
            alt={property.title}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            loading="lazy"
          />
          {property.property_type && (
            <span className="absolute top-2 left-2 bg-black/80 backdrop-blur-sm text-[10px] uppercase font-medium text-[#F5F5F5] px-2 py-0.5 rounded border border-[#27272A]">
              {property.property_type}
            </span>
          )}
        </div>
      ) : (
        <div className="h-20 w-full bg-surface-subtle flex items-center justify-center p-3 text-center border-b border-surface-border">
          <span className="text-xs text-[#A1A1AA] italic">Architecture & Development</span>
        </div>
      )}

      <div className="p-3.5 flex flex-col justify-between flex-grow">
        <div>
          <div className="flex items-center text-[#A1A1AA] text-[11px] mb-1">
            <MapPin className="w-3 h-3 mr-1 text-[#D4AF37] flex-shrink-0" />
            <span className="truncate">{property.location}</span>
          </div>

          <h4 className="text-xs font-medium text-[#F5F5F5] line-clamp-1 mb-1" title={property.title}>
            {property.title}
          </h4>

          {property.developer && (
            <p className="text-[10px] text-[#A1A1AA] mb-2">By {property.developer}</p>
          )}

          <div className="flex items-center justify-between text-xs my-2 pt-2 border-t border-surface-border">
            {property.price ? (
              <span className="font-medium text-[#F5F5F5]">{property.price}</span>
            ) : (
              <span className="text-[#A1A1AA] text-[11px]">Price on request</span>
            )}

            {property.bedrooms && (
              <span className="flex items-center text-[#A1A1AA] text-[11px]">
                <BedDouble className="w-3 h-3 mr-1 text-[#71717A]" />
                {property.bedrooms} Beds
              </span>
            )}
          </div>

          {property.amenities.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1 mb-2">
              {property.amenities.slice(0, 2).map((a, i) => (
                <span
                  key={i}
                  className="text-[9px] px-1.5 py-0.5 rounded bg-surface-subtle text-[#A1A1AA] border border-surface-border"
                >
                  {a}
                </span>
              ))}
            </div>
          )}
        </div>

        <a
          href={property.url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 text-center text-xs py-1.5 px-3 rounded-lg bg-surface-subtle hover:bg-[#1F1F23] border border-surface-border text-[#F5F5F5] flex items-center justify-center space-x-1.5 transition-colors"
        >
          <span>View Listing</span>
          <ExternalLink className="w-3 h-3 text-[#71717A]" />
        </a>
      </div>
    </div>
  );
};
