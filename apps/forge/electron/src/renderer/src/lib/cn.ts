import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Standard shadcn helper: `cn("p-2", isActive && "bg-accent")`.
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
